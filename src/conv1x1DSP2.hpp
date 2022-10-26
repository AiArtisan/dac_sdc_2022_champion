#ifndef __CONV1x1DS2_HPP__
#define __CONV1x1DS2_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

// #include "debug.hpp"
#include "function.h"
#include "stream_tools.h"

template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void streamInOneRowTwoPix(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][2][2]
                                      [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<1> rowBufferIdx) {
#pragma HLS inline off
  const unsigned INPENUM = SIMD / IN_PE;
  const unsigned SIMDNUM = IN_CH / SIMD;

  if (skip_flag)
    return;
  static ap_uint<IN_PE *IN_BIT> reg = 0;

  for (unsigned s = 0; s < SIMDNUM; s++)
    for (unsigned p = 0; p < INPENUM; p++)
      for (unsigned w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline
        ap_uint<IN_PE * IN_BIT> data1, data0;
        (data1, data0) = in.read();

        row_buffer[p][0][rowBufferIdx][w * SIMDNUM + s] = data0;
        row_buffer[p][1][rowBufferIdx][w * SIMDNUM + s] = data1;
      }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void streamOutOneRowTwoPix(
    stream<ap_uint<SIMD * IN_BIT>> &out,
    ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][2][2]
                                      [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<1> rowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete
  const unsigned IN_PE_BIT = IN_PE * IN_BIT;

  const unsigned INPENUM = SIMD / IN_PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  if (skip_flag)
    return;

  for (unsigned w = 0; w < IN_W; w++) {
    for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
      for (unsigned s = 0; s < SIMDNUM; s++) {
#pragma HLS pipeline
        ap_uint<SIMD * IN_BIT> data;
        ap_uint<IN_PE * IN_BIT> buffer_data[INPENUM];
#pragma HLS array_partition variable = buff_data complete
        ap_uint<1> sel = w % 2;

        for (unsigned i = 0; i < INPENUM; i++) {
          buffer_data[i] =
              row_buffer[i][sel][rowBufferIdx][w / 2 * SIMDNUM + s];
        }

        for (unsigned p = 0; p < INPENUM; p++) {
          data((p + 1) * IN_PE_BIT - 1, p * IN_PE_BIT) = buffer_data[p];
        }
        out.write(data);
      }
    }
  }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv1x1convert(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                    stream<ap_uint<SIMD * IN_BIT>> &out,
                    const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0");

  const unsigned INPENUM = SIMD / IN_PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  ap_uint<IN_PE * IN_BIT> row_buffer[INPENUM][2][2][IN_W / 2 * SIMDNUM];

#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
// #pragma HLS ARRAY_PARTITION variable = row_buffer dim = 2 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<1> storeBufferIdx = 0;
  ap_uint<1> loadBufferIdx = 1;

  for (unsigned rep = 0; rep < reps * IN_H + 1; rep++) {
#pragma HLS dependence intra false variable = row_buffer

    streamInOneRowTwoPix<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    streamOutOneRowTwoPix<IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, (rep == 0), loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;
  }
}

// template <unsigned W, unsigned CH, unsigned BIT, unsigned PE>
// void streamInOneRow(stream<ap_uint<PE * BIT>> &in,
//                     ap_uint<PE * BIT> row_buffer[W * CH / PE * 2],
//                     bool skip_flag, ap_uint<1> rowBufferIdx) {

//   const unsigned PENUM = CH / PE;

//   if (skip_flag)
//     return;

//   for (unsigned p = 0; p < PENUM; p++) {
//     for (unsigned w = 0; w < W; w++) {
// #pragma HLS pipeline
//       ap_uint<PE * BIT> data;
//       data = in.read();
//       row_buffer[(w * PENUM + p) * 2 + rowBufferIdx] = data;
//     }
//   }
// }

// template <unsigned W, unsigned CH, unsigned BIT, unsigned PE>
// void streamOutOneRow(stream<ap_uint<PE * BIT>> &out,
//                      ap_uint<PE * BIT> row_buffer[W * CH / PE * 2],
//                      bool skip_flag, ap_uint<1> rowBufferIdx) {
//   const unsigned PENUM = CH / PE;

//   if (skip_flag)
//     return;
//   unsigned addr = 0;

//   for (unsigned it = 0; it < W * PENUM; it++) {

// #pragma HLS pipeline
//     out << row_buffer[addr * 2 + rowBufferIdx];
//     addr++;
//   }
// }

// template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
// void reorderChannelPE(stream<ap_uint<PE * BIT>> &in,
//                       stream<ap_uint<PE * BIT>> &out, const unsigned reps = 1) {

//   const unsigned PENUM = CH / PE;
//   ap_uint<PE * BIT> row_buffer[COL * PENUM * 2];

//   ap_uint<1> storeBufferIdx = 0;
//   ap_uint<1> loadBufferIdx = 1;

//   for (unsigned rep = 0; rep < reps * ROW + 1; rep++) {
// #pragma HLS dependence intra false variable = row_buffer
// #pragma HLS dependence inter false variable = row_buffer
//     streamInOneRow<COL, CH, BIT, PE>(in, row_buffer, (rep >= reps * ROW),
//                                      storeBufferIdx);
//     streamOutOneRow<COL, CH, BIT, PE>(out, row_buffer, (rep == 0),
//                                       loadBufferIdx);
//     loadBufferIdx++;
//     storeBufferIdx++;
//   }
// }

template <unsigned IN_BIT, unsigned W_BIT, unsigned PROD_BIT, unsigned SIMD>
void simd_mac_DSP2(ap_uint<IN_BIT> invec[SIMD], ap_int<W_BIT> w0vec[SIMD],
                   ap_int<W_BIT> w1vec[SIMD], ap_int<PROD_BIT> &out0,
                   ap_int<PROD_BIT> &out1) {
#pragma HLS pipeline
#pragma HLS array_partition variable = invec
#pragma HLS array_partition variable = w1vec
#pragma HLS array_partition variable = w0vec
  ap_int<PROD_BIT * 2> acc = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_int<PROD_BIT + W_BIT + 1> rst = w1vec[i] * (1 << PROD_BIT) + w0vec[i];
    ap_int<PROD_BIT * 2> m = invec[i] * rst;
    acc += m;
  }

  out0 = acc(PROD_BIT - 1, 0);
  out1 = acc(PROD_BIT * 2 - 1, PROD_BIT) + acc[PROD_BIT - 1];
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned IN_BIT,
          unsigned IN_CH,

          unsigned W_BIT, unsigned B_BIT, unsigned M_BIT,

          unsigned SIMD, unsigned PE>
void conv1x1DSP2(stream<ap_uint<IN_BIT * SIMD>> &in,
                 const ap_uint<SIMD * W_BIT>
                     weights[PE][((IN_CH * 1) / SIMD) * (OUT_CH / PE)],
                 const ap_int<B_BIT> bias[PE][OUT_CH / PE],
                 stream<ap_uint<PE * M_BIT>> &out, const unsigned reps = 1) {
  const unsigned PROD_BIT = IN_BIT + W_BIT + 2;

#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  ap_int<W_BIT> wvec[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wvec complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wvec complete dim = 2

  ap_uint<IN_BIT> ivec[SIMD];
#pragma HLS ARRAY_PARTITION variable = ivec complete dim = 1

  ap_int<M_BIT> outPartialArr[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr complete dim = 1

  for (unsigned int h = 0; h < OUT_ROW * reps; h++) {
    for (unsigned int w = 0; w < OUT_COL; w++) {
      for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
        for (unsigned int simdIdx = 0; simdIdx < IN_CH / SIMD; simdIdx++) {
#pragma HLS pipeline
          ap_uint<IN_BIT *SIMD> inData = in.read();
          for (int s = 0; s < SIMD; s++) {
            ivec[s] = inData((s + 1) * IN_BIT - 1, s * IN_BIT);
          }
          for (int i = 0; i < PE; i++) {
            for (int s = 0; s < SIMD; s++) {
              wvec[i][s] = weights[i][peIdx * IN_CH / SIMD + simdIdx](
                  (s + 1) * W_BIT - 1, s * W_BIT);
            }
          }
          // cout << "w,kc:" << w << "," << kc << endl;

          for (int p = 0; p < PE; p += 2) {
            ap_int<PROD_BIT> outPartial0;
            ap_int<PROD_BIT> outPartial1;
            simd_mac_DSP2<IN_BIT, W_BIT, PROD_BIT, SIMD>(
                ivec, wvec[p], wvec[p + 1], outPartial0, outPartial1);
            if (simdIdx == 0) {
              outPartialArr[p] = outPartial0;
              outPartialArr[p + 1] = outPartial1;
            } else {
              outPartialArr[p] += outPartial0;
              outPartialArr[p + 1] += outPartial1;
            }
          }
          ap_uint<M_BIT * PE> odata;
          if (simdIdx == IN_CH / SIMD - 1) {
            for (int i = 0; i < PE; i++) {
              odata((i + 1) * M_BIT - 1, i * M_BIT) =
                  outPartialArr[i] + bias[i][peIdx];
            }
            out.write(odata);
          }
        }
    }
  }
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned OUT_CH, unsigned W_BIT, unsigned B_BIT, unsigned M_BIT,
          unsigned SIMD, unsigned PE, unsigned INPE>
void conv1x1_DSPopt(stream<ap_uint<IN_BIT * INPE * 2>> &in,
                    const ap_uint<SIMD * W_BIT>
                        weights[PE][((IN_CH * 1) / SIMD) * (OUT_CH / PE)],
                    const ap_int<B_BIT> bias[PE][OUT_CH / PE],
                    stream<ap_uint<PE * M_BIT>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW
  hls::stream<ap_uint<IN_BIT * SIMD>> conv1in("conv1in");
  conv1x1convert<IN_ROW, IN_COL, IN_CH, IN_BIT, INPE, SIMD, OUT_CH / PE>(
      in, conv1in, reps);

  conv1x1DSP2<IN_ROW, IN_COL, OUT_CH, IN_BIT, IN_CH, W_BIT, B_BIT, M_BIT, SIMD,
              PE>(conv1in, weights, bias, out, reps);
}

#endif