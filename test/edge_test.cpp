#include <random>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <gtest/gtest.h>
#include "../src/io.h"
#include "../src/edge.h"
#include "../src/move_search.h"
#include "test_boards.h"

namespace {

using mrand = std::uniform_int_distribution<size_t>;

const std::string kTestFile = "./io-test-file";
const std::string kTestIndexFile = kTestFile + ".index";

class EdgeTest : public ::testing::Test {
 protected:
  std::vector<PossibleMoves> moves;
  void SetUp() {
    for (auto& b : kTestBoards) {
      constexpr int piece = 0;
      auto board_map = b.PieceMap<piece>();
      auto m = MoveSearch<kLevel18, Board::NumRotations(piece), 18,
           0, 2, 2, 2, 2, 2, 2, 2, 2, 2>(board_map);
      moves.push_back(m);
    }
  }
  void TearDown() override {
    std::filesystem::remove(kTestFile);
    std::filesystem::remove(kTestIndexFile);
  }
};

EvaluateNodeEdges GenEvaluateEdges(const PossibleMoves& moves) {
  std::mt19937_64 gen;
  std::unordered_map<Position, int> mp;
  for (auto& i : moves.non_adj) mp.insert({i, (int)mp.size()});
  for (auto& adj : moves.adj) {
    for (auto& i : adj.second) mp.insert({i, (int)mp.size()});
  }
  std::vector<uint32_t> ind;
  for (size_t i = 0; i < mp.size(); i++) ind.push_back(i);
  std::shuffle(ind.begin(), ind.end(), gen);
  EvaluateNodeEdges ed;
  for (auto& i : ind) ed.next_ids.push_back({gen(), mrand(0, 4)(gen)});
  for (auto& i : moves.non_adj) ed.non_adj.push_back(ind[mp[i]]);
  for (auto& adj : moves.adj) {
    ed.adj.emplace_back();
    for (auto& i : adj.second) ed.adj.back().push_back(ind[mp[i]]);
  }
  return ed;
}

TEST_F(EdgeTest, Seek) {
}

} // namespace
