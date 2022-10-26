#ifndef __CONV2D_DSPOPT_HPP__
#define __CONV2D_DSPOPT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

// #include "debug.hpp"
#include "function.h"
#include "stream_tools.h"

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<2> rowBufferIdx) {
#pragma HLS inline off
  if (skip_flag)
    return;
  // ap_uint<IN_PE *IN_BIT> reg = 0;

  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)
    for (unsigned w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline
      ap_uint<IN_PE * IN_BIT * 2> data;
      ap_uint<IN_PE * IN_BIT> data0, data1;
      data = in.read();
      row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [w * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
    }
}


template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data(
    stream<ap_uint<SIMD * IN_BIT * 2 * K>> &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete
#pragma HLS array_partition variable = row_buffer dim = 2 complete


  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2;
  if (skip_flag)
    return;

  // ap_uint<8> infoldIdx = 0;
  ap_uint<5> simdIdx = 0;
  ap_uint<8> w = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned cycle = 0; cycle < WLEN * SIMDNUM; cycle++) {

#pragma HLS pipeline
      ap_uint<SIMD * IN_BIT> data0[K];
#pragma HLS array_partition variable = data0 dim = 0 complete
      ap_uint<SIMD * IN_BIT> data1[K];
#pragma HLS array_partition variable = data1 dim = 0 complete
      ap_uint<IN_PE * IN_BIT * 2> buffer_data[K][SIMD / IN_PE];
#pragma HLS array_partition variable = buff_data dim = 0 complete
      for (unsigned wr = 0; wr < K; wr++) {
#pragma HLS unroll
        ap_uint<2> rowBufferIdx = startRowBufferIdx + wr;
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
#pragma HLS unroll
          buffer_data[wr][i] = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];
        }

        if (outRowIdx - K / 2 + wr < 0 || outRowIdx - K / 2 + wr >= IN_H) {
          data0[wr] = 0;
          data1[wr] = 0;
        } else {
          for (unsigned i = 0; i < SIMD / IN_PE; i++) {
            data0[wr]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
                buffer_data[wr][i](IN_PE_BIT - 1, 0);
            data1[wr]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
                buffer_data[wr][i](IN_PE_BIT * 2 - 1, IN_PE_BIT);
          }
        }
      }

      out.write((data1[0], data0[0], data1[1], data0[1], data1[2], data0[2]));

      if (cycle == WLEN * SIMDNUM - 1) {
        w = 0;
      } else if (simdIdx == SIMDNUM - 1) {
        w++;
      }

      if (simdIdx == SIMDNUM - 1) {
        simdIdx = 0;
      } else {
        simdIdx++;
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                  stream<ap_uint<SIMD * IN_BIT * 2 * K>> &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0");
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                        [IN_W / 2 * IN_CH / SIMD];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 2 complete

#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;

  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, false, storeBufferIdx);
  storeBufferIdx++;
  
  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, false, storeBufferIdx);
  storeBufferIdx++;
  
  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, false, storeBufferIdx);
    stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, false, rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, false, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, false, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned PE>
void streamBnRelu(stream<ap_uint<PE * M_BIT * 2>> &in,
                  const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                  const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                  stream<ap_uint<PE * OUT_BIT * 2>> &out,
                  const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  ap_uint<PE * M_BIT> reg;
  ap_uint<PE * M_BIT * 2> data_in;
  ap_uint<PE * M_BIT> data0, data1;
  ap_uint<PE * OUT_BIT * 2> data_out;
  (data1, data0) = in.read();
  reg = data1;

  for (int r = 0; r < OUT_ROW * rep; r++)
    for (int peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (int w = 0; w < OUT_COL / 2; w++) {

#pragma HLS pipeline II = 2
        ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        (data1, data0) = in.read();
        data_in = (data0, reg);
        reg = data1;
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx],
                                       bias[i % PE][peIdx]);
        }
        out.write(data_out);
      }
}

template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS array_partition variable = ipack

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =
        (A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,
         B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] =
        (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}


template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE>
void simd_MAC(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
              ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_int<PROD_BIT + 5> &partial0, ap_int<PROD_BIT + 5> &partial1,
              ap_int<PROD_BIT + 5> &partial2, ap_int<PROD_BIT + 5> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  
  for (int i = 0; i < SIMD; i += CASCADE) {
#pragma HLS pipeline II = 1
#pragma HLS unroll 
    ap_int<PROD_BIT * 4> m = 0;
    for (int cs = 0; cs < CASCADE; cs++) {
#pragma HLS unroll
      m += wpack[i + cs] * ipack[i + cs];
    }

    ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT + 1> p1 = m(PROD_BIT * 2 - 1, PROD_BIT - 1);
    ap_int<PROD_BIT + 1> p2 = m(PROD_BIT * 3 - 1, PROD_BIT * 2 - 1);
    ap_int<PROD_BIT + 1> p3 = m(PROD_BIT * 4 - 1, PROD_BIT * 3 - 1);

    r0 += p0;
    r1 += (p1 >> 1) + (p1 & 1);
    r2 += (p2 >> 1) + (p2 & 1);
    r3 += (p3 >> 1) + (p3 & 1);

  }

  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}


template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT,
          unsigned OUT_W, unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned GUARD_BIT, unsigned M_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE,
          unsigned L_SHIFT>
void convDSPOpt(
    stream<ap_uint<SIMD * IN_BIT * 2 * K>> &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][K * K][IN_CH / SIMD * OUT_CH / PE],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    // stream<ap_uint<PE * OUT_BIT * 2>> &out,
    stream<ap_uint<PE * M_BIT * 2>> &out,
    const unsigned reps = 1) {

  static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT * 1;
  const unsigned OUT_WNUM = OUT_W / 2;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  ap_int<WPACK_BIT> wpacks0[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks0 complete dim = 2
  ap_int<WPACK_BIT> wpacks1[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks1 complete dim = 2
  ap_int<WPACK_BIT> wpacks2[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks2 complete dim = 2

  ap_uint<IPACK_BIT> ipack0[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack0 complete dim = 1
  ap_uint<IPACK_BIT> ipack1[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack1 complete dim = 1
  ap_uint<IPACK_BIT> ipack2[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack2 complete dim = 1

  ap_int<M_BIT> firPartialRes0[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes0 complete dim = 1
  ap_int<M_BIT> firPartialRes1[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes1 complete dim = 1

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  ap_int<PROD_BIT + 5> firPartial00;
  ap_int<PROD_BIT + 5> firPartial01;
  ap_int<PROD_BIT + 5> firPartial02;
  ap_int<PROD_BIT + 5> firPartial03;

  ap_int<PROD_BIT + 5> firPartial10;
  ap_int<PROD_BIT + 5> firPartial11;
  ap_int<PROD_BIT + 5> firPartial12;
  ap_int<PROD_BIT + 5> firPartial13;

  ap_int<PROD_BIT + 5> firPartial20;
  ap_int<PROD_BIT + 5> firPartial21;
  ap_int<PROD_BIT + 5> firPartial22;
  ap_int<PROD_BIT + 5> firPartial23;

  for (unsigned int h = 0; h < OUT_H * reps; h++) {
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int w = 0; w < OUT_WNUM; w++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
#pragma HLS pipeline 
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == SIMDNUM - 1);
          ap_uint<SIMD * IN_BIT> data1[K], data0[K];
#pragma HLS ARRAY_PARTITION variable = data0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = data1 complete dim = 1
          (data1[0], data0[0], data1[1], data0[1], data1[2], data0[2]) = vec.read();
          pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1[0], data0[0], ipack0);
          pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1[1], data0[1], ipack1);
          pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1[2], data0[2], ipack2);
          for (unsigned p = 0; p < PE; p++) {
#pragma HLS unroll
            pack_weight_data<W_BIT, SIMD, PROD_BIT>(
                weights[p][2][peIdx * SIMDNUM + infoldIdx],
                weights[p][1][peIdx * SIMDNUM + infoldIdx],
                weights[p][0][peIdx * SIMDNUM + infoldIdx], wpacks0[p]);
            pack_weight_data<W_BIT, SIMD, PROD_BIT>(
                weights[p][5][peIdx * SIMDNUM + infoldIdx],
                weights[p][4][peIdx * SIMDNUM + infoldIdx],
                weights[p][3][peIdx * SIMDNUM + infoldIdx], wpacks1[p]);
            pack_weight_data<W_BIT, SIMD, PROD_BIT>(
                weights[p][8][peIdx * SIMDNUM + infoldIdx],
                weights[p][7][peIdx * SIMDNUM + infoldIdx],
                weights[p][6][peIdx * SIMDNUM + infoldIdx], wpacks2[p]);
          }

          for (int p = 0; p < PE; p++) {
            // cout << "FIR result compare " << endl;
#pragma HLS unroll 

            simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE>(
                wpacks0[p], ipack0, firPartial00, firPartial01, firPartial02,
                firPartial03);
            
            simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE>(
                wpacks1[p], ipack1, firPartial10, firPartial11, firPartial12,
                firPartial13);
            
            simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE>(
                wpacks2[p], ipack2, firPartial20, firPartial21, firPartial22,
                firPartial23);
            // getchar();
            if (m_clear & o_clear) {
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial01 + firPartial11 + firPartial21;
            }
            if (m_clear & !o_clear) {
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial01 + firPartial11 + firPartial21;
            } 
            if (!m_clear & o_clear) {
              outPartialArr0[p] = firPartial00 + firPartial10 + firPartial20 + firPartialRes0[p];
              outPartialArr1[p] = firPartial01 + firPartial11 + firPartial21 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {
              outPartialArr0[p] += firPartial00 + firPartial10 + firPartial20;
              outPartialArr1[p] += firPartial01 + firPartial11 + firPartial21;
            }

            if (o_clear) {
              firPartialRes0[p] = firPartial02 + firPartial12 + firPartial22;
              firPartialRes1[p] = firPartial03 + firPartial13 + firPartial23;
            }
            else {
              firPartialRes0[p] += firPartial02 + firPartial12 + firPartial22;
              firPartialRes1[p] += firPartial03 + firPartial13 + firPartial23;
            }

          }

          if (o_out) {
            ap_uint<PE * M_BIT> out_buf0;
            ap_uint<PE * M_BIT> out_buf1;
            for (int p = 0; p < PE; p++) {
#pragma HLS unroll 
              out_buf0(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              out_buf1(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr1[p];
            }
            out.write((out_buf1, out_buf0));
          }
        }
      }
    }
  }
  ap_uint<PE * M_BIT> out_buf2;
  for (int p = 0; p < PE; p++) {
#pragma HLS unroll
    out_buf2(p * M_BIT + M_BIT - 1, p * M_BIT) = firPartialRes0[p];
  }
  out.write((0, out_buf2));
}


template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT,

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_bn_act_DSPopt(
    stream<ap_uint<IN_BIT * IN_PE * 2>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3 * 3]
                                       [(IN_CH / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  stream<ap_uint<SIMD * IN_BIT * 2 * 3>> padding_out("padding_out");
  conv3padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(
      in, padding_out, reps);

  stream<ap_uint<PE * OUT_BIT * 2>> mvau_out("mvau_out");
  stream<ap_uint<PE * M_BIT * 2>> conv_out("conv_out");
  convDSPOpt<3, IN_BIT, IN_CH, OUT_BIT, OUT_COL, OUT_ROW, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, CASCADE, PE, L_SHIFT>(
      padding_out, weights, inc, bias, conv_out, reps);
  
  streamBnRelu<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
                L_SHIFT, IN_BIT, W_BIT, PE>(conv_out, inc, bias, out,
                                            reps);
}
#endif
