#include "../board.h"
#include "../move_search.h"
#include "naive_functions.h"
#include "printing.h"
#include <gtest/gtest.h>

namespace {

class SearchTest : public ::testing::Test {
 protected:
  CompactBoard compactboard;

  void SetUp(long seed) {
    for (int i = 0; i < 25; i++) compactboard[i] = 0xff;
  }
  void TearDown() override {}
};

TEST_F(SearchTest, Level18) {
  SetUp(0);
  move_search::TapTable<0, 2, 2, 2, 2, 2, 2, 2, 2, 2> taps;
  {
    auto byte_map = GetPieceMap(Board(compactboard).ToByteBoard(), 0);
    auto board_map = Board(compactboard).PieceMap<0>();
    auto m1 = NaiveGetPossibleMoves(byte_map, kLevel18, 21, taps.data());
    auto m2 = Search<kLevel18, Board::NumRotations(0), 21, 0,
         2, 2, 2, 2, 2, 2, 2, 2, 2>().MoveSearch(board_map);
    m1.Normalize();
    m2.Normalize();
    // check separately for better printing
    EXPECT_EQ(m1.non_adj, m2.non_adj);
    EXPECT_EQ(m1.adj.size(), m2.adj.size());
    for (size_t i = 0; i < std::min(m1.adj.size(), m2.adj.size()); i++) {
      EXPECT_EQ(m1.adj[i].second.size(), m2.adj[i].second.size());
      EXPECT_EQ(m1.adj[i], m2.adj[i]);
    }
  }
}

} // namespace


