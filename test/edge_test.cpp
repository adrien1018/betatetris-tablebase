#include <random>
#include <algorithm>
#include <filesystem>
#include <unordered_set>
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
      auto m = MoveSearch<kLevel18, Board::NumRotations(piece), 18, Tap30Hz>(board_map);
      moves.push_back(m);
      m = MoveSearch<kLevel19, Board::NumRotations(piece), 18, Tap15Hz>(board_map);
      moves.push_back(m);
    }
  }
  void TearDown() override {}
};

std::pair<EvaluateNodeEdges, PositionNodeEdges> GenEdges(const PossibleMoves& moves) {
  std::mt19937_64 gen(0);
  std::unordered_map<Position, int> mp;
  for (auto& i : moves.non_adj) mp.insert({i, (int)mp.size()});
  for (auto& adj : moves.adj) {
    for (auto& i : adj.second) mp.insert({i, (int)mp.size()});
  }
  EvaluateNodeEdges ed_eval;
  PositionNodeEdges ed_pos;
  ed_pos.nexts.resize(mp.size());
  std::vector<uint32_t> ind;
  for (size_t i = 0; i < mp.size(); i++) {
    ind.push_back(i);
    ed_eval.next_ids.push_back({(uint32_t)gen(), mrand(0, 4)(gen)});
  }
  std::shuffle(ind.begin(), ind.end(), gen);
  for (auto& i : moves.non_adj) ed_eval.non_adj.push_back(ind[mp[i]]);
  for (auto& i : mp) ed_pos.nexts[ind[i.second]] = i.first;
  std::vector<Position> pos_adj;
  for (auto& adj : moves.adj) {
    ed_eval.adj.emplace_back();
    for (auto& i : adj.second) ed_eval.adj.back().push_back(ind[mp[i]]);
    pos_adj.push_back(adj.first);
  }
  auto adj_mp = ed_eval.ReduceAdj();
  for (auto& i : adj_mp) {
    ed_pos.adj.emplace_back();
    for (auto& j : i) ed_pos.adj.back().push_back(pos_adj[j]);
  }
  if (ed_pos.adj.size()) { // ensure multiple adj is tested
    ed_pos.adj[0].push_back({1, 1, 1});
    ed_pos.adj[0].push_back({1, 1, 2});
  }
  return {ed_eval, ed_pos};
}

TEST_F(EdgeTest, EvaluateSerialize) {
  for (auto& m : moves) {
    auto edges = GenEdges(m).first;
    std::vector<uint8_t> buf(edges.NumBytes());
    edges.GetBytes(buf.data());
    auto edges2 = EvaluateNodeEdges(buf.data(), buf.size());
    ASSERT_EQ(edges, edges2);

    edges.CalculateSubset();
    edges.use_subset = true;
    buf.resize(edges.NumBytes());
    edges.GetBytes(buf.data());
    edges.adj.clear();
    edges2 = EvaluateNodeEdges(buf.data(), buf.size());
    ASSERT_EQ(edges, edges2);
  }
}

TEST_F(EdgeTest, EvaluateSerializeFast) {
  for (auto& m : moves) {
    auto edges = GenEdges(m).first;
    std::vector<uint8_t> buf(edges.NumBytes());
    edges.GetBytes(buf.data());

    auto edges2 = EvaluateNodeEdgesFastTmpl<4096>(buf.data(), buf.size());
    ASSERT_EQ(edges2, edges);

    edges.CalculateSubset();
    edges.use_subset = true;
    buf.resize(edges.NumBytes());
    edges.GetBytes(buf.data());
    edges.adj.clear();
    edges2 = EvaluateNodeEdgesFastTmpl<4096>(buf.data(), buf.size());
    ASSERT_EQ(edges2, edges);
  }
}

TEST_F(EdgeTest, EvaluateSubset) {
  for (auto& m : moves) {
    auto edges = GenEdges(m).first;
    edges.CalculateSubset();
    std::vector<std::unordered_set<uint8_t>> s1;
    for (auto& i : edges.adj) s1.emplace_back(i.begin(), i.end());
    edges.CalculateAdj();
    std::vector<std::unordered_set<uint8_t>> s2;
    for (auto& i : edges.adj) s2.emplace_back(i.begin(), i.end());
    ASSERT_EQ(s1, s2);
  }
}

TEST_F(EdgeTest, PositionSerialize) {
  for (auto& m : moves) {
    auto edges = GenEdges(m).second;
    std::vector<uint8_t> buf(edges.NumBytes());
    edges.GetBytes(buf.data());
    auto edges2 = PositionNodeEdges(buf.data(), buf.size());
    ASSERT_EQ(edges, edges2);
  }
}

} // namespace
