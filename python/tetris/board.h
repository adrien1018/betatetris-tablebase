#pragma once

#include "../../src/board.h"
#include "python.h"

struct PythonBoard {
  PyObject_HEAD

  Board board;

  template <class... Args> PythonBoard(Args&&... args) : board(std::forward<Args>(args)...) {}
  operator Board() const { return board; }
};

extern PyTypeObject py_board_class;
