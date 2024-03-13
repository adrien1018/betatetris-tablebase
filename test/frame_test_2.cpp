#include "frame_test.h"

namespace {

TEST_F(FrameTest, Test12HzSmallAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    RandTest<kLevel18, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
    RandTest<kLevel19, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
    RandTest<kLevel29, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
    RandTest<kLevel39, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
  }
}

} // namespace
