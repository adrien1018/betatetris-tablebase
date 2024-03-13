#include "frame_test.h"

namespace {

TEST_F(FrameTest, TestTuck) {
  SetUp();
  for (auto& board : kTestTuckBoards) {
    PositionTest<kLevel18>(board);
    PositionTest<kLevel19>(board);
    PositionTest<kLevel29>(board);
    PositionTest<kLevel39>(board);
  }
}

} // namespace
