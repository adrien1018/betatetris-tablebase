#include "../src/board.h"
#include "../src/move_search.h"
#include "test_boards.h"
#include "naive_functions.h"
#include "printing.h"
#include <sstream>
#include <gtest/gtest.h>

namespace {

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {}
};

// template for loop
template<size_t N> struct Num { static const constexpr auto value = N; };
template <class F, std::size_t... Is>
void For(F func, std::index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(Num<Is>{}), 0)...};
}
template <std::size_t N, class Func>
void For(Func&& func) {
  For(func, std::make_index_sequence<N>());
}

template <Level level, int adj_delay, int... taps>
void TestSearch(const Board& b) {
  constexpr move_search::TapTable<taps...> taps_obj;
  For<7>([&](auto i_obj) {
    constexpr int piece = i_obj.value;
    constexpr Search<level, Board::NumRotations(piece), adj_delay, taps...> search{};
    auto byte_map = GetPieceMap(b.ToByteBoard(), piece);
    auto board_map = b.PieceMap<piece>();
    auto m1 = NaiveGetPossibleMoves(byte_map, level, adj_delay, taps_obj.data());
    auto m2 = search.MoveSearch(board_map);
    m1.Normalize();
    m2.Normalize();
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

TEST_F(SearchTest, Test30Hz) {
  SetUp();
  for (size_t i = 0; i < kTestBoards.size(); i++) {
    auto& board = kTestBoards[i];
    TestSearch<kLevel18, 21, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2>(board);
    TestSearch<kLevel19, 21, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2>(board);
    TestSearch<kLevel29, 21, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2>(board);
    TestSearch<kLevel39, 21, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2>(board);
    if (board.Count() >= 20) break;
  }
}

} // namespace


