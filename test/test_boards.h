#pragma once

#include "../src/board.h"
#include "../src/position.h"

extern const std::array<Board, 169> kTestBoards;
extern const std::array<Board, 164> kTestNoroBoards;

struct TestSearchBoard {
  Board board;
  int piece;
  Position pos;
  bool lvl_19_ok, lvl_39_ok;
};

#ifdef DOUBLE_TUCK
extern const std::array<TestSearchBoard, 28> kTestTuckBoards;
#else
extern const std::array<TestSearchBoard, 26> kTestTuckBoards;
#endif
