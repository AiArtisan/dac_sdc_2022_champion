#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace std;
#include "stream_tools.h"
#include <assert.h>

template <unsigned IN_BIT, unsigned SIMD, unsigned P>
void padding_var(
    stream<ap_uint<IN_BIT * SIMD> > &in, stream<ap_uint<IN_BIT * SIMD> > &out,
    const unsigned in_row,         //
    const unsigned in_col,         //
    const unsigned in_simd_pre_ch, // ch / simd
    const unsigned reps = 1) {
  // const unsigned OUT_ROW = in_row + 2 * P;
  const unsigned OUT_COL = in_col + 2 * P;
  // const unsigned DATA_NUM_PRE_CH = in_ch / SIMD;

  for (unsigned rep = 0; rep < reps; rep++) {

    for (unsigned h = 0; h < P; h++) {
      for (unsigned s = 0; s < OUT_COL; s++) {
        append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
      }
    }

    for (unsigned h = 0; h < in_row; h++) {

      for (unsigned s = 0; s < OUT_COL; s++) {
        // #pragma HLS PIPELINE II=1

        if ((s < P) || (s >= OUT_COL - P)) {
          // temp_out = 0;
          append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
        } else {
          // cout << "in size :" << in.size() << endl;
          stream_move<IN_BIT * SIMD>(in, out, in_simd_pre_ch);
        }
        // out.write(temp_out);
      }
    }

    for (unsigned h = 0; h < P; h++) {
      for (unsigned i = 0; i < OUT_COL; i++) {
        append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
      }
    }
  }
}


template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,
          unsigned P>
void padding(
    stream<ap_uint<IN_CH * IN_BIT> > &in, stream<ap_uint<IN_CH * IN_BIT> > &out,
    const unsigned reps = 1) {
  const unsigned OUT_ROW = IN_ROW + 2 * P;
  const unsigned OUT_COL = IN_COL + 2 * P;

  ap_uint<IN_CH *IN_BIT> temp_out = 0;

  for (unsigned rep = 0; rep < reps; rep++) {

    for (unsigned h = 0; h < P; h++) {
      for (unsigned s = 0; s < OUT_COL; s++) {
        out.write(0);
      }
    }

    for (unsigned h = 0; h < IN_ROW; h++) {

      for (unsigned s = 0; s < OUT_COL; s++) {
#pragma HLS PIPELINE II = 1

        if ((s < P) || (s >= OUT_COL - P)) {
          temp_out = 0;
        } else {
          temp_out = in.read();
        }

        out.write(temp_out);
      }
    }

    for (unsigned h = 0; h < P; h++) {
      for (unsigned i = 0; i < OUT_COL; i++) {
        out.write(0);
      }
    }
  }
}

// int d = 0;
template <unsigned IN_BIT, unsigned OUT_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT,

          unsigned DATA_BIT, unsigned W_BIT, unsigned L_SHIFT>
ap_uint<OUT_BIT> bn_qurelu(ap_int<IN_BIT> in, ap_int<INC_BIT> inc,
                           ap_int<BIAS_BIT> bias) {

  const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

  ap_int<IN_BIT> bn_res = in * inc + bias;
  ap_uint<OUT_BIT> res;

  if (bn_res > 0) {
    bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
    if (bn_res > 15) {
      res = 15;
    } else {
      res = bn_res;
    }
  } else {
    res = 0;
  }
  return res;
}

template <unsigned IN_BIT, unsigned OUT_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT,

          unsigned DATA_BIT, unsigned W_BIT, unsigned L_SHIFT>
ap_uint<OUT_BIT> bn_qurelu_fixed(ap_int<IN_BIT> in, ap_int<INC_BIT> inc,
                                 ap_int<BIAS_BIT> bias) {

  const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

  ap_int<IN_BIT + INC_BIT + 1> bn_res = in * inc + bias;
  ap_uint<OUT_BIT> res;

  if (bn_res > 0) {
    bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
    if (bn_res > 15) {
      res = 15;
    } else {
      res = bn_res;
    }
  } else {
    res = 0;
  }
  return res;
}
