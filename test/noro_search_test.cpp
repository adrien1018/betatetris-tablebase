#include <gtest/gtest.h>
#include "../src/move_search_noro.h"
#include "printing.h"
#include "test_boards.h"
#include "naive_functions.h"

class NoroSearchTest : public ::testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {}
};

namespace {

void TestSearch(const Board& b) {
  for (int taps_per_row = 0; taps_per_row <= 6; taps_per_row++) {
    for (size_t piece = 0; piece < kPieces; piece++) {
      for (bool do_tuck : {false, true}) {
        auto byte_map = GetPieceMap(b.ToByteBoard(), piece)[0];
        auto board_map = b.PieceMapNoro(piece);
        CompactBoard m1 = MoveSearchNoro(board_map, taps_per_row, do_tuck);
        CompactBoard m1_1 = MoveSearchNoro(b, piece, taps_per_row, do_tuck);
        ByteBoard m2 = NaiveNoroPossibleMoves(byte_map, taps_per_row, do_tuck);
        std::stringstream ss;
        using ::testing::PrintToString;
        ss << "{taps_per_row=" << taps_per_row << ",do_tuck=" << PrintToString(do_tuck)
            << ",piece=" << PrintToString(piece) << "}\n";
        PrintTo(b, &ss);
        const std::string info = ss.str();
        EXPECT_EQ(Board(m1), Board(m1_1)) << info;
        EXPECT_EQ(Board(m1), Board(m2)) << info;
      }
    }
  }
}

} // namespace

TEST_F(NoroSearchTest, TestNormalBoards) {
  SetUp();
  for (auto& board : kTestBoards) TestSearch(board);
}

TEST_F(NoroSearchTest, TestNoroBoards) {
  SetUp();
  for (auto& board : kTestNoroBoards) TestSearch(board);
}
