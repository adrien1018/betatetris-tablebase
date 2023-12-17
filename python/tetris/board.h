#pragma once

#include "../../src/board.h"
#include "python.h"

struct PythonBoard {
  PyObject_HEAD

  Board board;

  template <class... Args> PythonBoard(Args&&... args) : board(std::forward<Args>(args)...) {}
  operator Board() const { return board; }

  // a very simple heuristic to end early
  bool IsClean() const {
    int overhangs = board.NumOverhang();
    if (overhangs >= 4) return false;
    auto heights = board.ColumnHeights();
    int max_height = *std::max_element(heights.begin(), heights.end());
    if (overhangs >= 2) return max_height <= 3;
    if (overhangs >= 1) return max_height <= 5;
    if (max_height >= 12) return false;
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
