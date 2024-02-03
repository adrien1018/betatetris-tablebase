#include <random>
#include <gtest/gtest.h>
#include "../src/move_search.h"
#include "../src/frame_sequence.h"
#include "test_boards.h"
#include "printing.h"

namespace {

using mrand = std::uniform_int_distribution<int>;

class FrameTest : public ::testing::Test {
 protected:
  std::mt19937_64 gen;
  void SetUp() {}
  void TearDown() override {}
};

void AssertTapSep(const FrameSequence& seq, int frames) {
  frames = 2; // DOUBLE_TUCK
  int prev_ab = -100, prev_lr = -100;
  for (int i = 0; i < (int)seq.size(); i++) {
    if (seq[i].IsL() || seq[i].IsR()) {
      ASSERT_TRUE(i - prev_lr >= frames);
      prev_lr = i;
    }
    if (seq[i].IsA() || seq[i].IsB()) {
      ASSERT_TRUE(i - prev_ab >= frames);
      prev_ab = i;
    }
  }
}

template <Level level, int adj_delay, class Taps, class Rand>
void RandTest(const Board& b, int piece, int frame_sep, Rand& gen) {
  auto moves = MoveSearch<level, adj_delay, Taps>(b, piece);
  for (auto& i : moves.non_adj) {
    auto seq = GetFrameSequenceStart<level, Taps>(b, piece, adj_delay, i);
    ASSERT_TRUE(seq.size() >= adj_delay)
      << b << i << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
    AssertTapSep(seq, frame_sep);
    auto sim_pos = SimulateMove<level>(b, piece, seq, true);
    ASSERT_EQ(sim_pos, std::make_pair(i, true))
      << b << i << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
  }
  if (moves.adj.empty()) return;
  for (auto& i : moves.adj) {
    auto seq = GetFrameSequenceStart<level, Taps>(b, piece, adj_delay, i.first);
    ASSERT_TRUE(seq.size() >= adj_delay)
      << b << i.first << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
    AssertTapSep(seq, frame_sep);
    auto sim_pos = SimulateMove<level>(b, piece, seq, false);
    ASSERT_EQ(sim_pos, std::make_pair(i.first, false))
      << b << i.first << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
  }
  auto& adj = moves.adj[mrand(0, moves.adj.size() - 1)(gen)];
  auto pre_seq = GetFrameSequenceStart<level, Taps>(b, piece, adj_delay, adj.first);
  for (auto& i : adj.second) {
    auto seq = pre_seq;
    GetFrameSequenceAdj<level, Taps>(seq, b, piece, adj.first, i);
    AssertTapSep(seq, frame_sep);
    auto sim_pos = SimulateMove<level>(b, piece, seq, true);
    ASSERT_EQ(sim_pos, std::make_pair(i, true))
      << b << adj.first << i << ' ' << piece << ' ' << level << ' ' << seq << ' ' << adj_delay;
  }
}

template <Level level>
void PositionTest(const TestSearchBoard& b) {
  if (!b.lvl_39_ok && level == kLevel39) return;
  if (!b.lvl_19_ok && level != kLevel18) return;
  FrameSequence seq;
  std::pair<Position, bool> sim_pos;
  if constexpr (level == kLevel18 || level == kLevel19) {
    auto moves = MoveSearch<level, 18, Tap15Hz>(b.board, b.piece);
    for (auto& i : moves.adj) {
      if (std::find(i.second.begin(), i.second.end(), b.pos) == i.second.end()) continue;
      seq = GetFrameSequenceStart<level, Tap15Hz>(b.board, b.piece, 18, i.first);
      GetFrameSequenceAdj<level, Tap15Hz>(seq, b.board, b.piece, i.first, b.pos);
      sim_pos = SimulateMove<level>(b.board, b.piece, seq, true);
      ASSERT_EQ(sim_pos, std::make_pair(b.pos, true))
          << b.board << b.pos << ' ' << b.piece << ' ' << level << ' ' << seq;
    }
  }
  if constexpr (level == kLevel39) {
    seq = GetFrameSequenceStart<level, Tap30Hz>(b.board, b.piece, 18, b.pos);
    AssertTapSep(seq, 2);
  } else {
    seq = GetFrameSequenceStart<level, Tap15Hz>(b.board, b.piece, 61, b.pos);
    AssertTapSep(seq, 4);
  }
  sim_pos = SimulateMove<level>(b.board, b.piece, seq, true);
  ASSERT_EQ(sim_pos, std::make_pair(b.pos, true))
      << b.board << b.pos << ' ' << b.piece << ' ' << level << ' ' << seq;
}

TEST_F(FrameTest, Test30Hz) {
  SetUp();
  for (auto& board : kTestBoards) {
    RandTest<kLevel18, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
    RandTest<kLevel19, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
    RandTest<kLevel29, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
    RandTest<kLevel39, 18, Tap30Hz>(board, mrand(0, 6)(gen), 2, gen);
  }
}

TEST_F(FrameTest, Test12HzSmallAdj) {
  SetUp();
  for (auto& board : kTestBoards) {
    RandTest<kLevel18, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
    RandTest<kLevel19, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
    RandTest<kLevel29, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
    RandTest<kLevel39, 12, Tap12Hz>(board, mrand(0, 6)(gen), 5, gen);
  }
}

TEST_F(FrameTest, TestTuck) {
  SetUp();
  for (auto& board : kTestTuckBoards) {
    PositionTest<kLevel18>(board);
    PositionTest<kLevel19>(board);
    PositionTest<kLevel29>(board);
    PositionTest<kLevel39>(board);
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
