#pragma once

#include <cstdint>
#include <type_traits>
#include <immintrin.h>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

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
constexpr int popcount(T a) {
  static_assert(
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
#ifdef _MSC_VER
  return std::popcount(a);
#else
  return __builtin_popcountll(a);
#endif
}

template <class T>
constexpr int ctz(T a) {
  static_assert(
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
#ifdef _MSC_VER
  return std::countr_zero(a);
#else
  if constexpr(std::is_same<T, uint64_t>::value) {
    return __builtin_ctzll(a);
  } else {
    return __builtin_ctz(a);
  }
#endif
}

template <class T>
constexpr int clz(T a) {
  static_assert(
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, uint64_t>::value,
      "not implemented");
#ifdef _MSC_VER
  return std::countl_zero(a);
#else
  if constexpr(std::is_same<T, uint64_t>::value) {
    return __builtin_clzll(a);
  } else {
    return __builtin_clz(a);
  }
#endif
}

[[noreturn]] inline void unreachable() {
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#else
#ifdef _MSC_VER // MSVC
  __assume(false);
#endif
#endif
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

template <class T>
constexpr T abs(T x) {
  return x >= 0 ? x : -x;
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

#define DO_PIECE_CASE(func, b) \
  switch (piece) { \
    case 0: return func<level, Board::NumRotations(0) PIECE_CASE_TMPL_ARGS>(b.TMap() PIECE_CASE_ARGS); \
    case 1: return func<level, Board::NumRotations(1) PIECE_CASE_TMPL_ARGS>(b.JMap() PIECE_CASE_ARGS); \
    case 2: return func<level, Board::NumRotations(2) PIECE_CASE_TMPL_ARGS>(b.ZMap() PIECE_CASE_ARGS); \
    case 3: return func<level, Board::NumRotations(3) PIECE_CASE_TMPL_ARGS>(b.OMap() PIECE_CASE_ARGS); \
    case 4: return func<level, Board::NumRotations(4) PIECE_CASE_TMPL_ARGS>(b.SMap() PIECE_CASE_ARGS); \
    case 5: return func<level, Board::NumRotations(5) PIECE_CASE_TMPL_ARGS>(b.LMap() PIECE_CASE_ARGS); \
    case 6: return func<level, Board::NumRotations(6) PIECE_CASE_TMPL_ARGS>(b.IMap() PIECE_CASE_ARGS); \
  } \
  unreachable();
