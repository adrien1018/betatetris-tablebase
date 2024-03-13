#include "search_test.h"

namespace {

TEST_F(SearchTest, Test30Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<kLevel18, 18, Tap30Hz>(board);
    TestSearch<kLevel19, 18, Tap30Hz>(board);
    TestSearch<kLevel29, 18, Tap30Hz>(board);
    TestSearch<kLevel39, 18, Tap30Hz>(board);
  }
}

} // namespace
