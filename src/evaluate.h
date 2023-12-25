#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <immintrin.h>

class MoveEval {
 protected:
  static constexpr size_t kVecOutputSize = 7 * sizeof(float);

  static float Sum(__m256 vec) {
    alignas(32) float x[8];
    _mm256_store_ps(x, vec);
    float ret = 0;
    for (int i = 0; i < 8; i++) ret += x[i];
    return ret;
  }
 public:
  __m256 ev_vec;

  MoveEval() {}
  MoveEval(__m256 ev) : ev_vec(ev) {}
  MoveEval(const float ev[]) {
    LoadEv(ev);
  }
  MoveEval(const uint8_t buf[], size_t) {
    alignas(32) float ev[8] = {};
    memcpy(ev, buf, kVecOutputSize);
    ev_vec = _mm256_load_ps(ev);
  }

  static constexpr bool kIsConstSize = true;
  static constexpr size_t NumBytes() { return kVecOutputSize; }

  void GetBytes(uint8_t ret[]) const {
    GetEv(reinterpret_cast<float*>(ret));
  }

  void GetEv(float buf[]) const {
    alignas(32) float ret[8];
    _mm256_store_ps(ret, ev_vec);
    memcpy(buf, ret, kVecOutputSize);
  }

  void MaxWith(const MoveEval& x) {
    ev_vec = _mm256_max_ps(ev_vec, x.ev_vec);
  }

  __m256i MaxWithMask(const MoveEval& x) {
    __m256 mask = _mm256_cmp_ps(ev_vec, x.ev_vec, _CMP_LT_OQ);
    ev_vec = _mm256_blendv_ps(ev_vec, x.ev_vec, mask);
    return _mm256_castps_si256(mask);
  }

  __m256i MaxWithMask(const MoveEval& x, __m256i subst, int val) {
    __m256i mask = MaxWithMask(x);
    return _mm256_blendv_epi8(subst, _mm256_set1_epi32(val), mask);
  }

  void LoadEv(const float buf[]) {
    alignas(32) float x[8] = {};
    memcpy(x, buf, kVecOutputSize);
    ev_vec = _mm256_load_ps(x);
  }

  float Dot(__m256 probs) const {
    return Sum(_mm256_mul_ps(ev_vec, probs));
  }

  MoveEval& operator+=(float x) {
    ev_vec = _mm256_add_ps(ev_vec, _mm256_set1_ps(x));
    return *this;
  }
};

class NodeEval : public MoveEval {
  using MoveEval::kVecOutputSize;
 public:
  using MoveEval::ev_vec;
  __m256 var_vec;

  NodeEval() {}
  NodeEval(__m256 ev, __m256 var) : MoveEval(ev), var_vec(var) {}
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

  void GetVar(float buf[]) const {
    alignas(32) float ret[8];
    _mm256_store_ps(ret, var_vec);
    memcpy(buf, ret, kVecOutputSize);
  }

  void LoadVar(const float buf[]) {
    alignas(32) float x[8] = {};
    memcpy(x, buf, kVecOutputSize);
    var_vec = _mm256_load_ps(x);
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

std::vector<NodeEval> CalculatePiece(
    int pieces, const std::vector<NodeEval>& prev, const std::vector<size_t>& offsets);
std::vector<NodeEval> ReadValues(int pieces, size_t total_size = 0);
std::vector<MoveEval> ReadValuesEvOnly(int pieces, size_t total_size = 0);
void RunEvaluate(int start_pieces, const std::vector<int>& output_locations, bool sample);
