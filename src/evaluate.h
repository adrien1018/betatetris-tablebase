#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <immintrin.h>

class NodeEval {
  static constexpr size_t kVecOutputSize = 7 * sizeof(float);

  static float Sum(__m256 vec) {
    alignas(32) float x[8];
    _mm256_store_ps(x, vec);
    float ret = 0;
    for (int i = 0; i < 8; i++) ret += x[i];
    return ret;
  }
 public:
  __m256 ev_vec, var_vec;

  NodeEval() {}
  NodeEval(__m256 ev, __m256 var) : ev_vec(ev), var_vec(var) {}
  NodeEval(const float ev[], const float var[]) {
    LoadEv(ev);
    LoadVar(var);
  }
  NodeEval(const uint8_t buf[], size_t) {
    alignas(32) float ev[8] = {}, var[8] = {};
    memcpy(ev, buf, kVecOutputSize);
    memcpy(var, buf + kVecOutputSize, kVecOutputSize);
    ev_vec = _mm256_load_ps(ev);
    var_vec = _mm256_load_ps(var);
  }

  static constexpr bool kIsConstSize = true;
  static constexpr size_t NumBytes() { return kVecOutputSize * 2; }

  void GetBytes(uint8_t ret[]) const {
    GetEv(reinterpret_cast<float*>(ret));
    GetVar(reinterpret_cast<float*>(ret + kVecOutputSize));
  }

  void MaxWith(const NodeEval& x) {
    __m256 mask = _mm256_cmp_ps(ev_vec, x.ev_vec, _CMP_LT_OQ);
    ev_vec = _mm256_blendv_ps(ev_vec, x.ev_vec, mask);
    var_vec = _mm256_blendv_ps(var_vec, x.var_vec, mask);
  }

  void GetEv(float buf[]) const {
    alignas(32) float ret[8];
    _mm256_store_ps(ret, ev_vec);
    memcpy(buf, ret, kVecOutputSize);
  }

  void GetVar(float buf[]) const {
    alignas(32) float ret[8];
    _mm256_store_ps(ret, var_vec);
    memcpy(buf, ret, kVecOutputSize);
  }

  void LoadEv(const float buf[]) {
    alignas(32) float x[8] = {};
    memcpy(x, buf, kVecOutputSize);
    ev_vec = _mm256_load_ps(x);
  }

  void LoadVar(const float buf[]) {
    alignas(32) float x[8] = {};
    memcpy(x, buf, kVecOutputSize);
    var_vec = _mm256_load_ps(x);
  }

  float Dot(__m256 probs) const {
    return Sum(_mm256_mul_ps(ev_vec, probs));
  }

  float DotVar(__m256 probs, float fin_ev) const {
    // mixture distribution
    __m256 ei_minus_e = _mm256_sub_ps(ev_vec, _mm256_set1_ps(fin_ev));
    __m256 val = _mm256_fmadd_ps(ei_minus_e, ei_minus_e, var_vec);
    return Sum(_mm256_mul_ps(val, probs));
  }

  NodeEval& operator+=(float x) {
    ev_vec = _mm256_add_ps(ev_vec, _mm256_set1_ps(x));
    return *this;
  }
};

void RunEvaluate(int io_threads, int start_pieces, const std::vector<int>& output_locations);
