#include "printing.h"

#include <algorithm>

void PrintTo(const Position& x, std::ostream* os) {
  *os << '(' << x.r << ',' << x.x << ',' << x.y << ')';
}

void PrintTo(const Board& x, std::ostream* os) {
  *os << x.ToString();
}
