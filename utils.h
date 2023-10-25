#pragma once

#include "board.h"

#include <cstdio>
#include <vector>

void PrintBoard(const Board& b) {
  for (auto& row : b.ToByteBoard()) {
    for (int i = 0; i < 10; i++) printf("%d%c", (int)row[i], " \n"[i == 9]);
  }
}

template <size_t R> void PrintBoards(const std::array<Board, R>& bs) {
  std::vector<ByteBoard> byte_bs;
  for (auto& b : bs) byte_bs.emplace_back(b.ToByteBoard());
  for (int row = 0; row < 20; row++) {
    for (size_t r = 0; r < R; r++) {
      for (int col = 0; col < 10; col++) {
        printf("%d", (int)byte_bs[r][row][col]);
        if (col != 9) putchar(' ');
      }
      if (r == R - 1) puts("");
      else printf(" | ");
    }
  }
}
