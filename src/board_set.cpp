#include "board_set.h"

#include <fstream>
#include <algorithm>

#include "io.h"
#include "files.h"
#include "config.h"
#include "thread_pool.hpp"

void SplitBoards(const std::filesystem::path& fname) {
  std::array<std::vector<CompactBoard>, 5> boards;
  ClassReader<CompactBoard> reader(fname);
  size_t num_boards = 0;
  try {
    num_boards = BoardCount(fname);
  } catch (std::exception&) {}
  for (auto& i : boards) i.reserve(num_boards * 0.22 + 1);
  while (true) {
    auto chunk = reader.ReadBatch(65536);
    if (chunk.empty()) break;
    for (auto& i : chunk) boards[i.Group()].push_back(i);
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
  size_t kBlock = 65536;
  for (size_t i = 0;; i += kBlock) {
    auto chunk = reader.ReadBatch(kBlock);
    for (size_t j = 0; j < chunk.size(); j++) ret[Board(chunk[j])] = i + j;
    if (chunk.size() < kBlock) break;
  }
  return ret;
}

void BuildEdges() {
  BS::thread_pool pool(kParallel);
  std::array<BoardMap, 5> maps;
  pool.parallelize_loop(0, 5, [&](int l, int r){
    for (int i = l; i < r; i++) maps[i] = GetBoardMap(i);
  }).wait();

}
