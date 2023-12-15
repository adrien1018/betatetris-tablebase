#include <random>
#include <gtest/gtest.h>
#include "test_boards.h"
#include "naive_functions.h"

namespace {

constexpr int kSeedMax = 1000;

class BoardTest : public ::testing::Test {
 protected:
  ByteBoard byteboard;

  void SetUp(float density_l, float density_r, long seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0, 1);
    float density = dist(gen) * (density_r - density_l) + density_l;
    for (int i = 0; i < 20; i++) {
      for (int j = 0; j < 10; j++) byteboard[i][j] = dist(gen) < density ? 0 : 1;
    }
  }
  void TearDown() override {}
};

TEST_F(BoardTest, ByteBoardConvert) {
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0, 1, seed);
    Board board(byteboard);
    ASSERT_EQ(byteboard, board.ToByteBoard());
  }
}

TEST_F(BoardTest, CompactBoardConvert) {
  uint8_t buf[25];
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0, 1, seed);
    Board board(byteboard);
    board.ToBytes(buf);
    ASSERT_EQ(board, Board(buf));
  }
}

TEST_F(BoardTest, BoardCount) {
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0, 1, seed);
    Board board(byteboard);
    int cnt = 0;
    for (int i = 0; i < 20; i++) {
      for (int j = 0; j < 10; j++) cnt += !byteboard[i][j];
    }
    ASSERT_EQ(board.Count(), cnt);
  }
}

TEST_F(BoardTest, BoardColumnAndCell) {
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0, 1, seed);
    Board board(byteboard);
    for (int i = 0; i < 10; i++) {
      auto col = board.Column(i);
      for (int j = 0; j < 20; j++) {
        ASSERT_EQ(col >> j & 1, byteboard[j][i]);
        ASSERT_EQ(board.Cell(j, i), byteboard[j][i]);
      }
    }
  }
}

TEST_F(BoardTest, LineClear) {
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0.3, 1, seed);
    Board board(byteboard);
    auto [lines_1, board_1] = board.ClearLines();
    auto board_2 = byteboard;
    int lines_2 = ClearLines(board_2);
    ASSERT_EQ(lines_1, lines_2);
    ASSERT_EQ(board_1, Board(board_2));
  }
}

TEST_F(BoardTest, NumOverhang) {
  for (auto& board : kTestBoards) {
    auto byteboard = board.ToByteBoard();
    ASSERT_EQ(board.NumOverhang(), NumOverhang(byteboard));
  }
}

TEST_F(BoardTest, ColumnHeights) {
  for (auto& board : kTestBoards) {
    auto byteboard = board.ToByteBoard();
    ASSERT_EQ(board.ColumnHeights(), ColumnHeights(byteboard));
  }
}

class BoardTestParam : public BoardTest, public testing::WithParamInterface<int> {};
TEST_P(BoardTestParam, TestBoardMap) {
  int piece = GetParam();
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0, 0.9, seed);
    Board board(byteboard);
    auto map_board = board.PieceMap(piece);
    auto map_byteboard = GetPieceMap(byteboard, piece);
    // speed test
    //auto map_byteboard = Board(board.ToBytes().data()).PieceMap(piece);
    ASSERT_EQ(map_board.size(), map_byteboard.size());
    for (size_t i = 0; i < map_board.size(); i++) {
      ASSERT_EQ(map_board[i].ToByteBoard(), map_byteboard[i]);
      ASSERT_EQ(map_board[i], Board(map_byteboard[i])); // check for normalization
    }
  }
}

TEST_P(BoardTestParam, TestPiecePlace) {
  int piece = GetParam();
  for (int seed = 0; seed < kSeedMax; seed++) {
    SetUp(0, 0.9, seed);
    Board board(byteboard);
    auto map_board = board.PieceMap(piece);
    for (size_t r = 0; r < map_board.size(); r++) {
      for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) {
          if (!map_board[r].Cell(i, j)) continue;
          auto place_board = board.Place(piece, r, i, j);
          auto place_byteboard = PlacePiece(byteboard, piece, r, i, j);
          // speed test
          //auto place_byteboard = Board(board.ToBytes().data()).Place(piece, r, i, j);
          ASSERT_EQ(place_board.ToByteBoard(), place_byteboard);
          ASSERT_EQ(place_board, Board(place_byteboard)); // check for normalization
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Board, BoardTestParam,
    testing::Values(0, 1, 2, 3, 4, 5, 6));

} // namespace
