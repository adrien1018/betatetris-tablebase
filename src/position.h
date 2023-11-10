#pragma once

#include <tuple>
#include "hash.h"

struct Position {
  static constexpr bool kIsConstSize = true;

  int r, x, y;
  Position L() const { return {r, x, y - 1}; }
  Position R() const { return {r, x, y + 1}; }
  Position D() const { return {r, x + 1, y}; }
  template <int R> Position A() const { return {(r + 1) % R, x, y}; }
  template <int R> Position B() const { return {(r + R - 1) % R, x, y}; }

  constexpr auto operator<=>(const Position&) const = default;

  static constexpr size_t NumBytes() { return 2; }
  void GetBytes(uint8_t ret[]) const {
    ret[0] = r << 5 | x;
    ret[1] = y;
  }
  static Position FromBytes(const uint8_t data[], size_t) {
    return {data[0] >> 5, data[0] & 31, data[1]};
  }
};

namespace std {

template<>
struct hash<Position> {
  constexpr size_t operator()(const Position& p) const {
    return Hash(p.r, p.x*16+p.y);
  }
};

} // namespace std
