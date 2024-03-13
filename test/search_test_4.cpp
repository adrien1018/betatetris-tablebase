#include "search_test.h"

namespace {

TEST_F(SearchTest, Test12Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<kLevel18, 21, Tap12Hz>(board);
    TestSearch<kLevel19, 21, Tap12Hz>(board);
    TestSearch<kLevel29, 21, Tap12Hz>(board);
    TestSearch<kLevel39, 21, Tap12Hz>(board);
  }
}

} // namespace
