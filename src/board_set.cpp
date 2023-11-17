#include "board_set.h"

#include <fstream>
#include <algorithm>
#include <tsl/hopscotch_map.h>

#include "edge.h"
#include "config.h"
#include "move_search.h"
#include "thread_queue.h"

namespace {

constexpr size_t kBlock = 65536;

} // namespace

void SplitBoards(const std::filesystem::path& fname) {
  std::array<std::vector<CompactBoard>, kGroups> boards;
  ClassReader<CompactBoard> reader(fname);
  size_t num_boards = 0;
  try {
    num_boards = BoardCount(fname);
  } catch (std::exception&) {}
  for (auto& i : boards) i.reserve(num_boards * 0.22 + 1);
  while (true) {
    auto chunk = reader.ReadBatch(kBlock);
    for (auto& i : chunk) boards[i.Group()].push_back(i);
    if (chunk.size() < kBlock) break;
  }
  BS::thread_pool pool(kParallel);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int i = l; i < r; i++) {
      std::sort(boards[i].begin(), boards[i].end(), [](const CompactBoard& a, const CompactBoard& b){
        return a.Count() == b.Count() ? a < b : a.Count() < b.Count();
      });
    }
  }).wait();
  for (int group = 0; group < kGroups; group++) {
    ClassWriter<CompactBoard> writer(BoardPath(group));
    writer.Write(boards[group]);
  }
}

BoardMap GetBoardMap(int group) {
  auto fname = BoardPath(group);
  size_t num_boards = BoardCount(fname);

  BoardMap ret;
  ret.reserve(num_boards);
  size_t i = 0;
  ProcessBoards(group, [&](Board&& b) { ret[b] = i++; });
  return ret;
}

namespace {

using TapSpeed = TAP_SPEED;
constexpr int kAdjDelay = ADJ_DELAY;

inline std::pair<EvaluateNodeEdges, PositionNodeEdges> GetEdges(
    const Board& b, int piece, const PossibleMoves& moves, const BoardMap& mp) {
  constexpr uint64_t kNone = std::numeric_limits<uint64_t>::max();
  tsl::hopscotch_map<Position, std::pair<uint64_t, uint8_t>> mp_next;
  mp_next.reserve(48);
  for (auto& pos : moves.non_adj) mp_next[pos] = {kNone, 0};
  for (auto& adj : moves.adj) {
    for (auto& pos : adj.second) mp_next[pos] = {kNone, 0};
  }
  for (auto item = mp_next.begin(); item != mp_next.end();) {
    auto n_board = b.Place(piece, item->first.r, item->first.x, item->first.y).ClearLines();
    if (auto board_it = mp.find(n_board.second); board_it != mp.end()) {
      item.value() = {board_it->second, n_board.first};
      ++item;
    } else {
      item = mp_next.erase(item);
    }
  }
  tsl::hopscotch_map<Position, uint8_t> mp_idx;
  EvaluateNodeEdges eval_ed;
  PositionNodeEdges pos_ed;
  eval_ed.cell_count = b.Count();
  { // nexts
    uint8_t idx = 0;
    mp_idx.reserve(mp_next.size());
    eval_ed.next_ids.reserve(mp_next.size());
    pos_ed.nexts.reserve(mp_next.size());
    for (auto& item : mp_next) {
      eval_ed.next_ids.push_back(item.second);
      pos_ed.nexts.push_back(item.first);
      mp_idx[item.first] = idx++;
    }
  }
  // non-adjs
  for (auto& pos : moves.non_adj) {
    if (auto it = mp_idx.find(pos); it != mp_idx.end()) {
      eval_ed.non_adj.push_back(it->second);
    }
  }
  // adjs
  for (auto& adj : moves.adj) {
    std::vector<uint8_t> ids;
    for (auto& pos : adj.second) {
      if (auto it = mp_idx.find(pos); it != mp_idx.end()) {
        ids.push_back(it->second);
      }
    }
    if (ids.empty()) continue;
    eval_ed.adj.push_back(std::move(ids));
    pos_ed.adj.push_back(adj.first);
    ids.clear();
  }
  return {eval_ed, pos_ed};
}

using EdgeChunk = std::pair<std::array<std::vector<EvaluateNodeEdges>, kLevels>,
                            std::array<std::vector<PositionNodeEdges>, kLevels>>;

EdgeChunk BuildEdgeChunk(const std::vector<Board>&& boards, const BoardMap& mp) {
  EdgeChunk ret;
  auto& [eval_eds, pos_eds] = ret;
  std::array<std::vector<std::array<PossibleMoves, kPieces>>, kLevels> search_results;
  for (int i = 0; i < kLevels; i++) search_results[i].resize(boards.size());
  For<kLevels>([&](auto level_obj) {
    constexpr int level_val = level_obj.value;
    constexpr Level level = static_cast<Level>(level_obj.value);
    for (size_t i = 0; i < boards.size(); i++) {
      search_results[level_val][i][0] = MoveSearch<level, 4, ADJ_DELAY, TAP_SPEED>(boards[i].TMap());
      search_results[level_val][i][1] = MoveSearch<level, 4, ADJ_DELAY, TAP_SPEED>(boards[i].JMap());
      search_results[level_val][i][5] = MoveSearch<level, 4, ADJ_DELAY, TAP_SPEED>(boards[i].LMap());
    }
    for (size_t i = 0; i < boards.size(); i++) {
      search_results[level_val][i][2] = MoveSearch<level, 2, ADJ_DELAY, TAP_SPEED>(boards[i].ZMap());
      search_results[level_val][i][4] = MoveSearch<level, 2, ADJ_DELAY, TAP_SPEED>(boards[i].SMap());
      search_results[level_val][i][6] = MoveSearch<level, 2, ADJ_DELAY, TAP_SPEED>(boards[i].IMap());
    }
    for (size_t i = 0; i < boards.size(); i++) {
      search_results[level_val][i][3] = MoveSearch<level, 1, ADJ_DELAY, TAP_SPEED>(boards[i].OMap());
    }
  });
  for (int level = 0; level < kLevels; level++) {
    auto& cur_eval = eval_eds[level];
    auto& cur_pos = pos_eds[level];
    auto& cur_moves = search_results[level];
    cur_eval.reserve(boards.size() * kPieces);
    cur_pos.reserve(boards.size() * kPieces);
    for (size_t i = 0; i < boards.size(); i++) {
      for (size_t j = 0; j < kPieces; j++) {
        auto [eval_ed, pos_ed] = GetEdges(boards[i], j, cur_moves[i][j], mp);
        cur_eval.push_back(std::move(eval_ed));
        cur_pos.push_back(std::move(pos_ed));
      }
    }
  }
  return ret;
}

void BuildEdges(int group) {
  int nxt_group = (group + 2) % kGroups;
  BoardMap mp = GetBoardMap(nxt_group);

  std::vector<ClassWriter<EvaluateNodeEdges>> eval_writers;
  std::vector<ClassWriter<PositionNodeEdges>> pos_writers;
  for (int level = 0; level < kLevels; level++) {
    eval_writers.emplace_back(EvaluateEdgePath(group, level), 128);
    pos_writers.emplace_back(PositionEdgePath(group, level), 256);
  }

  auto thread_queue = MakeThreadQueue<EdgeChunk>(kParallel, [&](EdgeChunk&& chunk) {
    for (int level = 0; level < kLevels; level++) {
      eval_writers[level].Write(chunk.first[level]);
      pos_writers[level].Write(chunk.second[level]);
    }
  });
  std::vector<Board> block;
  block.reserve(1024);
  ProcessBoards(group, [&](Board&& b) {
    block.push_back(std::move(b));
    if (block.size() == 1024) {
      thread_queue.Push([block,&mp]() { return BuildEdgeChunk(std::move(block), std::cref(mp)); });
      block.clear();
    }
  });
  thread_queue.Push([block,&mp]() { return BuildEdgeChunk(std::move(block), std::cref(mp)); });
  // destructor will finish the jobs
}

} // namespace

void BuildEdges() {
  std::array<BoardMap, kGroups> maps;
  for (int i = 0; i < kGroups; i++) BuildEdges(i);
}
