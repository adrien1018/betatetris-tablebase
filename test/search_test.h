#include <sstream>
#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/move_search.h"
#include "naive_functions.h"
#include "test_boards.h"
#include "printing.h"

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {}
};

template <Level level, int adj_delay, class Taps>
void TestSearch(const Board& b) {
  constexpr Taps taps_obj;
  For<7>([&](auto i_obj) {
    constexpr int piece = i_obj.value;
    auto byte_map = GetPieceMap(b.ToByteBoard(), piece);
    auto board_map = b.PieceMap<piece>();
    auto m1 = NaiveGetPossibleMoves(byte_map, level, adj_delay, taps_obj.data());
    auto m2 = MoveSearch<level, Board::NumRotations(piece), adj_delay, Taps>(board_map);
    m1.Normalize();
    size_t old_sz = m2.non_adj.size() + m2.adj.size();
    for (auto& i : m2.adj) old_sz += i.second.size();
    m2.Normalize();
    size_t new_sz = m2.non_adj.size() + m2.adj.size();
    for (auto& i : m2.adj) new_sz += i.second.size();
    EXPECT_EQ(old_sz, new_sz);
    std::stringstream ss;
    using ::testing::PrintToString;
    ss << "{level=" << PrintToString(level) << ",adj_delay=" << PrintToString(adj_delay) << ",piece=" << PrintToString(piece) << "}\n";
    PrintTo(b, &ss);
    const std::string info = ss.str();
    // check separately for better printing
    EXPECT_EQ(m1.non_adj, m2.non_adj) << info;
    EXPECT_EQ(m1.adj.size(), m2.adj.size()) << info;
    for (size_t i = 0; i < std::min(m1.adj.size(), m2.adj.size()); i++) {
      EXPECT_EQ(m1.adj[i].second.size(), m2.adj[i].second.size()) << info;
      EXPECT_EQ(m1.adj[i], m2.adj[i]) << info;
    }
  });
}

template <Level level>
void TestSearchPosition(const TestSearchBoard& b) {
  PossibleMoves moves;
  if constexpr (level == kLevel18 || level == kLevel19) {
    bool expected = level == kLevel18 || b.lvl_19_ok;
    moves = MoveSearch<level, 18, Tap20Hz>(b.board, b.piece);
    EXPECT_EQ(std::any_of(moves.adj.begin(), moves.adj.end(), [&b](const auto& q) {
            return std::find(q.second.begin(), q.second.end(), b.pos) != q.second.end();
          }), expected) << b.board.ToString() << "adj," << (int)level;
  }
  if constexpr (level == kLevel39) {
    moves = MoveSearch<level, 18, Tap30Hz>(b.board, b.piece);
  } else {
    moves = MoveSearch<level, 61, Tap20Hz>(b.board, b.piece);
  }
  bool expected = level == kLevel18 || (b.lvl_19_ok && (level != kLevel39 || b.lvl_39_ok));
  EXPECT_EQ(std::find(moves.non_adj.begin(), moves.non_adj.end(), b.pos) != moves.non_adj.end(), expected)
      << b.board.ToString() << "non_adj," << (int)level;
}

template <Level level, class Taps>
void TestZeroAdj(const Board& b) {
  constexpr Taps taps_obj;
  For<7>([&](auto i_obj) {
    constexpr int piece = i_obj.value;
    auto byte_map = GetPieceMap(b.ToByteBoard(), piece);
    auto board_map = b.PieceMap<piece>();
    auto m = MoveSearch<level, Board::NumRotations(piece), 0, Taps>(board_map);
    m.Normalize();
    auto m_noadj = MoveSearch<level, Board::NumRotations(piece), 61, Taps>(board_map);
    m_noadj.Normalize();

    std::stringstream ss;
    using ::testing::PrintToString;
    ss << "{level=" << PrintToString(level) << ",piece=" << PrintToString(piece) << "}\n";
    PrintTo(b, &ss);
    const std::string info = ss.str();
    // check separately for better printing
    bool has_start = false;
    for (size_t i = 0; i < m.adj.size(); i++) {
      if (m.adj[i].first.r == Position::Start.r && m.adj[i].first.y == Position::Start.y) {
        EXPECT_EQ(m_noadj.non_adj, m.adj[i].second) << info << m.adj[i].first;
        has_start = true;
        break;
      }
    }
    EXPECT_TRUE(has_start || m_noadj.non_adj.empty()) << info;
  });
}
