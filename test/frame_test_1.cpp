#include "frame_test.h"

namespace {

TEST_F(FrameTest, Test30Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    RandTest<kLevel18, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
    RandTest<kLevel19, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
    RandTest<kLevel29, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
    RandTest<kLevel39, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
  }
}

TEST_F(FrameTest, TestBestAdj) {
  SetUp();
  Board b = Board::Ones;
  auto moves = MoveSearch<kLevel18, 18, Tap30Hz>(b, 0);
  {
    std::array<Position, 7> pos{{{2, 19, 3}, {2, 19, 3}, {2, 19, 3}, {2, 19, 3},
                                 {2, 19, 5}, {2, 19, 5}, {2, 19, 5}}};
    size_t idx = GetBestAdj<kLevel18, Tap30Hz>(b, 0, moves, 18, pos.data()).first;
    ASSERT_EQ(moves.adj[idx].first, Position(2, 6, 4));
  } {
    std::array<Position, 7> pos{{{2, 19, 3}, {2, 19, 5}, {2, 19, 5}, {2, 19, 5},
                                 {2, 19, 5}, {2, 19, 5}, {2, 19, 5}}};
    size_t idx = GetBestAdj<kLevel18, Tap30Hz>(b, 0, moves, 18, pos.data()).first;
    ASSERT_EQ(moves.adj[idx].first, Position(2, 6, 5));
  } {
    std::array<Position, 7> pos{{{0, 18, 5}, {0, 18, 5}, {0, 18, 5}, {0, 18, 5},
                                 {2, 19, 5}, {2, 19, 5}, {2, 19, 5}}};
    size_t idx = GetBestAdj<kLevel18, Tap30Hz>(b, 0, moves, 18, pos.data()).first;
    ASSERT_TRUE(moves.adj[idx].first == Position(1, 6, 5) ||
                moves.adj[idx].first == Position(3, 6, 5));
  }
  using namespace std::literals;
  b = Board("....X.....\n"
            ".....X...."sv);
  moves = MoveSearch<kLevel18, 18, Tap30Hz>(b, 0);
  {
    std::array<Position, 7> pos{{{2, 19, 3}, {2, 19, 3}, {2, 19, 3}, {2, 19, 3},
                                 {2, 19, 3}, {2, 19, 3}, {2, 19, 3}}};
    size_t idx = GetBestAdj<kLevel18, Tap30Hz>(b, 0, moves, 18, pos.data()).first;
    ASSERT_EQ(moves.adj[idx].first, Position(2, 6, 2));
  }
}

} // namespace
