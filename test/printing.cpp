#include "printing.h"

#include <algorithm>

void PrintTo(const Position& x, std::ostream* os) {
  *os << '(' << x.r << ',' << x.x << ',' << x.y << ')';
}

void PrintTo(const Board& x, std::ostream* os) {
  auto b = x.ToByteBoard();
  int first_row = 0;
  while (first_row < 20 && std::all_of(b[first_row].begin(), b[first_row].end(), [](auto x){return x;})) first_row++;
  if (first_row > 0) first_row--;
  for (int row = first_row; row < 20; row++) {
    std::string str = std::to_string(row);
    if (str.size() == 1) str = ' ' + str;
    *os << str << ' ';
    for (auto i : b[row]) *os << "X."[i];
    *os << '\n';
  }
}
