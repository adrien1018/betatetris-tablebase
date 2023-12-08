#pragma once

#include <cstdint>
#include <type_traits>
#include <immintrin.h>

template <class T>
constexpr T pext(T a, T mask) {
  static_assert(
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
  if (std::is_constant_evaluated()) {
    T res = 0;
    for (T bb = 1; mask != 0; bb <<= 1) {
      if (a & mask & -mask) res |= bb;
      mask &= (mask - 1);
    }
    return res;
  } else if constexpr(std::is_same<T, uint64_t>::value) {
    return _pext_u64(a, mask);
  } else {
    return _pext_u32(a, mask);
  }
}

template <class T>
constexpr T pdep(T a, T mask) {
  static_assert(
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
  if (std::is_constant_evaluated()) {
    T res = 0;
    for (T bb = 1; mask; bb <<= 1) {
      if (a & bb) res |= mask & -mask;
      mask &= mask - 1;
    }
    return res;
  } else if constexpr(std::is_same<T, uint64_t>::value) {
    return _pdep_u64(a, mask);
  } else {
    return _pdep_u32(a, mask);
  }
}

template <class T>
constexpr T BytesToInt(const uint8_t x[]) {
  static_assert(
      std::is_same<T, uint16_t>::value ||
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
}
template <class T>
constexpr void IntToBytes(T x, uint8_t ret[]) {
  static_assert(
      std::is_same<T, uint16_t>::value ||
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
}

template <>
constexpr uint64_t BytesToInt(const uint8_t x[]) {
  using U = uint64_t;
  // compiler only optimized it if expanded out like this
  return (U)x[0]       | (U)x[1] << 8  | (U)x[2] << 16 | (U)x[3] << 24 |
         (U)x[4] << 32 | (U)x[5] << 40 | (U)x[6] << 48 | (U)x[7] << 56;
}

template <>
constexpr void IntToBytes(uint64_t x, uint8_t ret[]) {
  ret[0] = x;       ret[1] = x >> 8;  ret[2] = x >> 16; ret[3] = x >> 24;
  ret[4] = x >> 32; ret[5] = x >> 40; ret[6] = x >> 48; ret[7] = x >> 56;
}

template <>
constexpr uint32_t BytesToInt(const uint8_t x[]) {
  using U = uint32_t;
  return (U)x[0] | (U)x[1] << 8 | (U)x[2] << 16 | (U)x[3] << 24;
}

template <>
constexpr void IntToBytes(uint32_t x, uint8_t ret[]) {
  ret[0] = x; ret[1] = x >> 8; ret[2] = x >> 16; ret[3] = x >> 24;
}

template <>
constexpr uint16_t BytesToInt(const uint8_t x[]) {
  using U = uint16_t;
  return (U)x[0] | (U)x[1] << 8;
}

template <>
constexpr void IntToBytes(uint16_t x, uint8_t ret[]) {
  ret[0] = x; ret[1] = x >> 8;
}

// constexpr loop
template<size_t N>
struct TemplateNum { static const constexpr auto value = N; };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
template <class F, std::size_t... Is>
void For(F func, std::index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(TemplateNum<Is>{}), 0)...};
}
#pragma GCC diagnostic pop
template <std::size_t N, class Func>
void For(Func&& func) {
  For(func, std::make_index_sequence<N>());
}
