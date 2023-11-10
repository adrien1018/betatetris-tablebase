#include "board_set.h"

#include <fstream>
#include <algorithm>

#include "io.h"
#include "files.h"
#include "config.h"
#include "move_search.h"
#include "thread_pool.hpp"

namespace {

constexpr size_t kBlock = 65536;

} // namespace

void SplitBoards(const std::filesystem::path& fname) {
  std::array<std::vector<CompactBoard>, 5> boards;
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
  pool.parallelize_loop(0, 5, [&](int l, int r){
    for (int i = l; i < r; i++) {
      std::sort(boards[i].begin(), boards[i].end(), [](const CompactBoard& a, const CompactBoard& b){
        return a.Count() == b.Count() ? a < b : a.Count() < b.Count();
      });
    }
  }).wait();
  for (size_t group = 0; group < 5; group++) {
    ClassWriter<CompactBoard> writer(BoardPath(group));
    writer.Write(boards[group]);
  }
}

BoardMap GetBoardMap(int group) {
  auto fname = BoardPath(group);
  ClassReader<CompactBoard> reader(fname);
  size_t num_boards = BoardCount(fname);

  BoardMap ret;
  ret.reserve(num_boards);
  for (size_t i = 0;; i += kBlock) {
    auto chunk = reader.ReadBatch(kBlock);
    for (size_t j = 0; j < chunk.size(); j++) ret[Board(chunk[j])] = i + j;
    if (chunk.size() < kBlock) break;
  }
  return ret;
}

std::vector<Board> GetBoards(int group) {
  auto fname = BoardPath(group);
  ClassReader<CompactBoard> reader(fname);
  size_t num_boards = BoardCount(fname);

  std::vector<Board> ret;
  ret.reserve(num_boards);
  while (true) {
    auto chunk = reader.ReadBatch(kBlock);
    for (auto& i : chunk) ret.push_back(Board(i));
    if (chunk.size() < kBlock) break;
  }
  return ret;
}

namespace {

using TapSpeed = TAP_SPEED;
constexpr int kAdjDelay = ADJ_DELAY;

void BuildEdgesChunk(const std::vector<Board>& boards, const BoardMap& mp) {
  std::array<std::vector<std::array<PossibleMoves, 7>>, 4> search_results;
  For<4>([&](auto level_obj) {
    constexpr int level_val = level_obj.value;
    constexpr Level level = (Level)level_obj.value;
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
}

void BuildEdges(int group) {
  int nxt_group = (group + 2) % 5;
  BoardMap mp = GetBoardMap(nxt_group);
}

} // namespace

void BuildEdges() {
  BS::thread_pool pool(kParallel);
  std::array<BoardMap, 5> maps;
  pool.parallelize_loop(0, 5, [&](int l, int r){
    for (int i = l; i < r; i++) maps[i] = GetBoardMap(i);
  }).wait();

}
