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

TEST_F(SearchTest, Test30HzSmallAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<kLevel18, 4, Tap30Hz>(board);
    TestSearch<kLevel19, 4, Tap30Hz>(board);
    TestSearch<kLevel29, 4, Tap30Hz>(board);
    TestSearch<kLevel39, 4, Tap30Hz>(board);
  }
}

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

TEST_F(SearchTest, Test12Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    TestSearch<kLevel18, 21, Tap12Hz>(board);
    TestSearch<kLevel19, 21, Tap12Hz>(board);
    TestSearch<kLevel29, 21, Tap12Hz>(board);
    TestSearch<kLevel39, 21, Tap12Hz>(board);
  }
}

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
