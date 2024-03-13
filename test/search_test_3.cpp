#include "search_test.h"

namespace {

TEST_F(SearchTest, Test30HzZeroAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<kLevel18, 0, Tap30Hz>(board);
    TestSearch<kLevel19, 0, Tap30Hz>(board);
    TestSearch<kLevel29, 0, Tap30Hz>(board);
    TestSearch<kLevel39, 0, Tap30Hz>(board);
    TestZeroAdj<kLevel18, Tap30Hz>(board);
    TestZeroAdj<kLevel19, Tap30Hz>(board);
    TestZeroAdj<kLevel29, Tap30Hz>(board);
    TestZeroAdj<kLevel39, Tap30Hz>(board);
  }
}

} // namespace
