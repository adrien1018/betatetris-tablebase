#pragma once

#include "board.h"

constexpr CompactBoard MoveSearchNoro(const Board& b, int inputs_per_row, bool do_tuck) {
  auto rows = b.Rows();
  CompactBoard result{};
  auto WriteResultRow = [&result,&rows](int row, uint32_t state) {
    uint32_t lock = row == 19 ? state : (state & ~rows[row + 1]);
    int offset = row * 10 / 8, bit = row * 10 % 8;
    result[offset] |= lock << bit;
    result[offset + 1] |= lock >> (8 - bit);
  };

  if (do_tuck && inputs_per_row) {
    uint32_t state = 1 << 5;
    for (int row = 0; row < 20 && state; row++) {
      state &= rows[row];
      for (int i = 0; i < inputs_per_row; i++) {
        state |= (state << 1 | state >> 1) & rows[row];
        if (row == 0 && i == 0) state |= rows[row] & (1 << 4 | 1 << 6);
      }
      WriteResultRow(row, state);
    }
  } else if (do_tuck && !inputs_per_row) { // 29
    uint32_t state0 = 1 << 5, state1 = 1 << 5;
    for (int row = 0; row < 20 && (state0 || state1); row++) {
      state0 &= rows[row];
      state1 &= rows[row];
      uint32_t nstate0 = state0 | state1;
      state1 = (state0 << 1 | state0 >> 1) & rows[row];
      if (row == 0) state1 |= rows[row] & (1 << 4 | 1 << 6);
      state0 = nstate0;
      WriteResultRow(row, state0 | state1);
    }
  } else {
    uint32_t state = 1 << 5;
    bool left = state, right = state;
    for (int row = 0; row < 20 && state; row++) {
      state &= rows[row];
      int nl = inputs_per_row ? row * inputs_per_row : (row + 1) / 2;
      int nr = inputs_per_row ? (row + 1) * inputs_per_row : (row + 2) / 2;
      if (nl <= 5 && !(1 << (5 - nl) & rows[row])) left = false;
      if (nl <= 4 && !(1 << (5 + nl) & rows[row])) right = false;
      for (int i = nl + 1; i <= nr && i <= 5; i++) {
        if ((left || i == 1) && i <= 5 && (1 << (5 - i) & rows[row])) {
          state |= 1 << (5 - i);
          left = true;
        } else {
          left = false;
        }
        if ((right || i == 1) && i <= 4 && (1 << (5 + i) & rows[row])) {
          state |= 1 << (5 + i);
          right = true;
        } else {
          right = false;
        }
      }
      WriteResultRow(row, state);
    }
  }
  return result;
}

constexpr CompactBoard MoveSearchNoro(const Board& b, int piece, int inputs_per_row, bool do_tuck) {
  return MoveSearchNoro(b.PieceMapNoro(piece), inputs_per_row, do_tuck);
}
