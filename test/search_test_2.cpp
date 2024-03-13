#include "search_test.h"

namespace {

TEST_F(SearchTest, Test30HzSmallAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<kLevel18, 4, Tap30Hz>(board);
    TestSearch<kLevel19, 4, Tap30Hz>(board);
    TestSearch<kLevel29, 4, Tap30Hz>(board);
    TestSearch<kLevel39, 4, Tap30Hz>(board);
  }
}

} // namespace
