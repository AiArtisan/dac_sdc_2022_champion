#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#include <ap_int.h>
#include <fstream>
#include <hls_stream.h>
#include <iomanip>
#include <string>

using namespace std;
template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned OUT_BIT>
void print_FM_stream(hls::stream<ap_uint<OUT_BIT * OUT_CH>> &in,
                     string filename) {
  ofstream f(filename);

  for (int r = 0; r < OUT_ROW; r++)
    for (int c = 0; c < OUT_COL; c++) {
      f.width(10);
      f << '[' << r << "," << c << "]";
      ap_uint<OUT_BIT *OUT_CH> data = in.read();

      for (int d = 0; d < OUT_CH; d++) {
        ap_int<OUT_BIT> wdata = data(d * OUT_BIT + OUT_BIT - 1, d * OUT_BIT);
        f << wdata;
      }
      f << endl;
    }
  f.close();
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned OUT_BIT>
void print_FM_stream_through(hls::stream<ap_uint<OUT_BIT * OUT_CH>> &out,
                             string filename) {
  ofstream f(filename);

  for (int r = 0; r < OUT_ROW; r++)
    for (int c = 0; c < OUT_COL; c++) {
      f << '[' << setw(4) << r << "," << setw(4) << c << "]";
      ap_uint<OUT_BIT *OUT_CH> data = out.read();
      out.write(data);
      for (int d = 0; d < OUT_CH; d++) {
        ap_uint<OUT_BIT> wdata = data(d * OUT_BIT + OUT_BIT - 1, d * OUT_BIT);
        f << wdata << ",";
      }
      f << endl;
    }
  f.close();
}

template <unsigned K, unsigned ROW, unsigned COL, unsigned CH, unsigned SIMD,
          unsigned BIT, unsigned PENUM>
void print_SWU_stream_through(hls::stream<ap_uint<BIT * SIMD * 2>> &out,
                              string filename) {
  ofstream f(filename);
  for (int r = 0; r < ROW; r++)
    for (int peIdx = 0; peIdx < PENUM; peIdx++)
      for (int c = 0; c < COL + K - 1; c += 2) {
        for (int kh = 0; kh < K; kh++)
          for (int cs = 0; cs < CH / SIMD; cs++) {
            f << '[' << setw(3) << r + kh << "," << setw(3) << c << ","
              << setw(3) << cs << "]";
            ap_uint<BIT *SIMD * 2> data = out.read();
            // out.write(data);
            for (int d = 0; d < BIT * SIMD * 2 / 8; d++) {
              ap_uint<8> wdata = data(d * 8 + 7, d * 8);
              f << setw(4) << wdata;
            }
            f << endl;
          }
      }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PENUM, unsigned BIT>
void print_l0_padding_stream_through(hls::stream<ap_uint<BIT * CH * 3>> &out,
                                     string filename) {
  ofstream f(filename);
  for (int r = 0; r < ROW; r++)
    for (int peIdx = 0; peIdx < PENUM; peIdx++)
      for (int c = 0; c < COL; c++) {
        for (int kc = -1; kc < 2; kc++) {
          f << '[' << setw(3) << r << "," << setw(3) << c + kc << "," << setw(3)
            << peIdx << "]";
          ap_uint<BIT *CH * 3> data = out.read();
          // out.write(data);
          for (int d = 0; d < BIT * CH * 3 / 8; d++) {
            ap_uint<8> wdata = data(d * 8 + 7, d * 8);
            f << setw(4) << wdata;
          }
          f << endl;
        }
      }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PENUM, unsigned BIT,
          unsigned SIMD>
void print_conv1x1_through(hls::stream<ap_uint<BIT * SIMD>> &in,
                           string filename) {
  ofstream f(filename);
  for (int r = 0; r < ROW; r++)
    for (int c = 0; c < COL; c++) {
      for (int peIdx = 0; peIdx < PENUM; peIdx++)
        for (int ch = 0; ch < CH; ch += SIMD) {
          f << '[' << setw(3) << r << "," << setw(3) << c << "," << setw(3)
            << peIdx << "]";
          ap_uint<BIT *CH> data = in.read();
          // out.write(data);
          for (int d = 0; d < SIMD; d++) {
            ap_uint<8> wdata = data(d * BIT + BIT - 1, d * BIT);
            f << setw(4) << wdata;
          }
          f << endl;
        }
    }

  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_mavu_DSPopt_stream_through(hls::stream<ap_uint<BIT * PE * 2>> &out,
                                      string filename, unsigned reps) {
  ofstream f(filename);
  ap_uint<BIT * PE> buffer[CH / PE][COL];

  for (int r = 0; r < ROW * reps; r++) {
    for (int peIdx = 0; peIdx < CH / PE; peIdx++) {
      for (int c = 0; c < COL; c += 2) {
        ap_uint<BIT *PE * 2> data = out.read();
        out << data;
        buffer[peIdx][c] = data(BIT * PE - 1, 0);
        buffer[peIdx][c + 1] = data(BIT * PE * 2 - 1, BIT * PE);
      }
    }
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      for (int peIdx = 0; peIdx < CH / PE; peIdx++) {
        for (int p = 0; p < PE; p++) {
          ap_uint<BIT> data =
              buffer[peIdx][c].range(p * BIT + BIT - 1, p * BIT);
          f << data.to_string(16) << ",";
        }
      }
      f << endl;
    }
  }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_output_featuremap(ap_uint<BIT> OFM[CH][ROW][COL], string filename,
                             unsigned reps) {
  ofstream f(filename);
  ap_uint<BIT * PE> buffer[CH / PE][COL];

  for (int r = 0; r < ROW * reps; r++) {
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      for (int ch = 0; ch < CH; ch++) {
        f << OFM[ch][r][c].to_string(16) << ",";
      }
      f << endl;
    }
  }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_output_featuremap_signed(ap_int<BIT> OFM[CH][ROW][COL],
                                    string filename, unsigned reps) {
  ofstream f(filename);
  ap_uint<BIT * PE> buffer[CH / PE][COL];

  for (int r = 0; r < ROW * reps; r++) {
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      for (int ch = 0; ch < CH; ch++) {
        f << OFM[ch][r][c].to_string(10) << ",";
      }
      f << endl;
    }
  }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_mavu_stream_through(hls::stream<ap_uint<BIT * CH>> &out,
                               string filename, unsigned reps) {
  ofstream f(filename);

  for (int r = 0; r < ROW * reps; r++) {
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      ap_uint<BIT *CH> outdata = out.read();
      out << outdata;
      for (int ch = 0; ch < CH; ch++) {
        ap_uint<BIT> data = outdata.range(ch * BIT + BIT - 1, ch * BIT);
        f << data.to_string(16) << ",";
      }
      f << endl;
    }
  }
  f.close();
}

template <unsigned ROW, unsigned COL, unsigned CH, unsigned PE, unsigned BIT>
void print_pe_stream_through(hls::stream<ap_uint<BIT * PE>> &out,
                             string filename, unsigned reps) {
  ofstream f(filename);

  for (int r = 0; r < ROW * reps; r++) {
    for (int c = 0; c < COL; c++) {
      f << "[" << setw(4) << r << "," << setw(4) << c << "]";
      for (int pp = 0; pp < CH; pp += PE) {
        ap_uint<BIT *PE> outdata = out.read();
        out << outdata;

        for (int p = 0; p < PE; p++) {
          ap_int<BIT> data = outdata.range(p * BIT + BIT - 1, p * BIT);
          f << data.to_string(10) << ",";
        }
      }
      f << endl;
    }
  }
  f.close();
}

// template <unsigned ROW, unsigned COL, unsigned CH, unsigned BIT>
// void load_featuremap(string filename, ap_uint<BIT> IFM[CH][ROW][COL],
//                      float factor) {
//   float *IFM_float = new float[ROW * COL * CH];

//   FILE *fp = fopen(filename.c_str(), "rb");

//   size_t ret = fread(IFM_float, sizeof(float), ROW * COL * CH, fp);
//   assert(ret == ROW * COL * CH);

//   for (int ch = 0; ch < CH; ch++)
//     for (int r = 0; r < ROW; r++) {
//       for (int c = 0; c < COL; c++) {

//         float rst = round(IFM_float[ch * ROW * COL + r * COL + c] * factor);
//         // assert(rst <= factor && rst >= 0);
//         IFM[ch][r][c] = rst;
//       }
//     }
//   delete[] IFM_float;
//   fclose(fp);
// }

// template <unsigned ROW, unsigned COL, unsigned CH, unsigned BIT>
// void load_featuremap_signed(string filename, ap_int<BIT> IFM[CH][ROW][COL],
//                             float factor) {
//   float *IFM_float = new float[ROW * COL * CH];

//   FILE *fp = fopen(filename.c_str(), "rb");

//   size_t ret = fread(IFM_float, sizeof(float), ROW * COL * CH, fp);
//   assert(ret == ROW * COL * CH);

//   for (int ch = 0; ch < CH; ch++)
//     for (int r = 0; r < ROW; r++) {
//       for (int c = 0; c < COL; c++) {

//         float rst = round(IFM_float[ch * ROW * COL + r * COL + c] * factor);
//         // assert(rst <= factor && rst >= 0);
//         IFM[ch][r][c] = rst;
//       }
//     }
//   delete[] IFM_float;
//   fclose(fp);
// }
#endif
