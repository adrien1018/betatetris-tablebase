#include "printing.h"

#include <algorithm>

void PrintTo(const Position& x, std::ostream* os) {
  *os << '(' << x.r << ',' << x.x << ',' << x.y << ')';
}

void PrintTo(const Board& x, std::ostream* os) {
  *os << x.ToString();
}

void PrintTo(const FrameInput& x, std::ostream* os) {
  std::string str;
  if (x.IsL()) str += 'L';
  if (x.IsR()) str += 'R';
  if (x.IsA()) str += 'A';
  if (x.IsB()) str += 'B';
  if (str.empty()) str += '-';
  *os << str;
}

void PrintTo(const FrameSequence& x, std::ostream* os) {
  if (x.empty()) {
    *os << "(empty seq)";
  }
  for (size_t i = 0; i < x.size(); i++) {
    PrintTo(x[i], os);
    if (i + 1 != x.size()) *os << ' ';
  }
}
