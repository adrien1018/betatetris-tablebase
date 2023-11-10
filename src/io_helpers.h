#pragma once

#include <cstdint>
#include <vector>
#include "constexpr_helpers.h"

template <size_t size_bytes, class T, class Func>
inline size_t VecOutput(const std::vector<T>& vec, uint8_t data[], Func&& func) {
  uint8_t sz_buf[8] = {};
  IntToBytes<uint64_t>(vec.size(), sz_buf);
  memcpy(data, sz_buf, size_bytes);
  size_t ind = size_bytes;
  for (auto& i : vec) ind += func(i, data + ind);
  return ind;
}

template <size_t size_bytes, class T, class Func>
inline size_t VecInput(std::vector<T>& vec, const uint8_t data[], Func&& func) {
  uint8_t sz_buf[8] = {};
  memcpy(sz_buf, data, size_bytes);
  vec.resize(BytesToInt<uint64_t>(sz_buf));
  size_t ind = size_bytes;
  for (auto& i : vec) ind += func(i, data + ind);
  return ind;
}

template <size_t size_bytes, class T>
inline size_t SimpleVecOutput(const std::vector<T>& vec, uint8_t data[]) {
  uint8_t sz_buf[8] = {};
  IntToBytes<uint64_t>(vec.size(), sz_buf);
  memcpy(data, sz_buf, size_bytes);
  memcpy(data + size_bytes, vec.data(), sizeof(T) * vec.size());
  return sizeof(T) * vec.size() + size_bytes;
}

template <size_t size_bytes, class T>
inline size_t SimpleVecInput(std::vector<T>& vec, const uint8_t data[]) {
  uint8_t sz_buf[8] = {};
  memcpy(sz_buf, data, size_bytes);
  vec.resize(BytesToInt<uint64_t>(sz_buf));
  memcpy(vec.data(), data + size_bytes, sizeof(T) * vec.size());
  return sizeof(T) * vec.size() + size_bytes;
}

template <class T, size_t sz>
struct SimpleIOArray : public std::array<T, sz> {
  using std::array<T, sz>::array;
  using std::array<T, sz>::data;

  SimpleIOArray(const uint8_t buf[], size_t) {
    memcpy(data(), buf, NumBytes());
  }

  static constexpr bool kIsConstSize = true;
  static constexpr size_t NumBytes() { return sizeof(T) * sz; }
  void GetBytes(uint8_t ret[]) const {
    memcpy(ret, data(), NumBytes());
  }
};
