#include "search_test.h"

namespace {

TEST_F(SearchTest, TestTuck) {
  SetUp();
  for (auto& board : kTestTuckBoards) {
    TestSearchPosition<kLevel18>(board);
    TestSearchPosition<kLevel19>(board);
    TestSearchPosition<kLevel29>(board);
    TestSearchPosition<kLevel39>(board);
  }
}

} // namespace
