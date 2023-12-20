#pragma once

#include "../src/board.h"
#include "../src/position.h"

extern const std::array<Board, 169> kTestBoards;

struct TestSearchBoard {
  Board board;
  int piece;
  Position pos;
  bool lvl_39_ok;
};

extern const std::array<TestSearchBoard, 24> kTestTuckBoards;
