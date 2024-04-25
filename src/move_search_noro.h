#pragma once

#include "board.h"

constexpr CompactBoard MoveSearchNoro(const Board& b, int inputs_per_row, bool do_tuck) {
  auto rows = b.Rows();
  CompactBoard result{};
  if (do_tuck) {
    uint32_t state = 1 << 5;
    for (int row = 0; row < 20 && state; row++) {
      state &= rows[row];
      for (int i = 0; i < inputs_per_row; i++) {
        state |= (state << 1 | state >> 1) & rows[row];
      }
      uint32_t lock = row == 19 ? state : (state & ~rows[row + 1]);
      int offset = row * 10 / 8, bit = row * 10 % 8;
      result[offset] |= lock << bit;
      result[offset + 1] |= lock >> (8 - bit);
    }
  } else {
    uint32_t state = 1 << 5;
    bool left = state, right = state;
    for (int row = 0; row < 20 && state; row++) {
      state &= rows[row];
      int ns = row * inputs_per_row;
      if (ns <= 5 && !(1 << (5 - ns) & rows[row])) left = false;
      if (ns <= 4 && !(1 << (5 + ns) & rows[row])) right = false;
      for (int i = ns + 1; (left || right) && i <= ns + inputs_per_row; i++) {
        if (left && i <= 5 && (1 << (5 - i) & rows[row])) {
          state |= 1 << (5 - i);
        } else {
          left = false;
        }
        if (right && i <= 4 && (1 << (5 + i) & rows[row])) {
          state |= 1 << (5 + i);
        } else {
          right = false;
        }
      }
      uint32_t lock = row == 19 ? state : (state & ~rows[row + 1]);
      int offset = row * 10 / 8, bit = row * 10 % 8;
      result[offset] |= lock << bit;
      result[offset + 1] |= lock >> (8 - bit);
    }
  }
  return result;
}

constexpr CompactBoard MoveSearchNoro(const Board& b, int piece, int inputs_per_row, bool do_tuck) {
  return MoveSearchNoro(b.PieceMapNoro(piece), inputs_per_row, do_tuck);
}
