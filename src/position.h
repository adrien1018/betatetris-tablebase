#pragma once

#include <tuple>

struct Position {
  int r, x, y;
  Position L() const { return {r, x, y - 1}; }
  Position R() const { return {r, x, y + 1}; }
  Position D() const { return {r, x + 1, y}; }
  template <int R> Position A() const { return {r == R - 1 ? 0 : r + 1, x, y}; }
  template <int R> Position B() const { return {r == 0 ? R - 1 : r - 1, x, y}; }
  bool operator<(const Position& p) const { return std::tie(r, x, y) < std::tie(p.r, p.x, p.y); }
  bool operator==(const Position& p) const { return std::tie(r, x, y) == std::tie(p.r, p.x, p.y); }
};
