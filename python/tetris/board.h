#pragma once

#include <algorithm>
#include "python.h"
#include "../../src/board.h"

struct PythonBoard {
  PyObject_HEAD

  Board board;

  template <class... Args> PythonBoard(Args&&... args) : board(std::forward<Args>(args)...) {}
  operator Board() const { return board; }

  // a very simple heuristic to end early
  bool IsClean() const {
    int overhangs = board.NumOverhang();
    if (overhangs >= 1) return false;
    auto heights = board.ColumnHeights();
    int differences = 0, well = 0;
    for (int i = 0; i < 9; i++) differences += abs(heights[i] - heights[i+1]);
    if (heights[0] == 0) well = std::max(well, heights[1]);
    if (heights[9] == 0) well = std::max(well, heights[8]);
    for (int i = 1; i < 9; i++) {
      if (heights[i] == 0) well = std::max(well, heights[i-1] + heights[i+1]);
    }
    return differences - well <= 10;
  }
};

extern PyTypeObject py_board_class;
