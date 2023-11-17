#pragma once

#include <vector>
#include <filesystem>
#include <tsl/sparse_map.h>

#include "io.h"
#include "board.h"
#include "files.h"

using BoardMapKey = std::pair<Board, uint64_t>;
using BoardMap = tsl::sparse_map<Board, uint64_t, std::hash<Board>, std::equal_to<Board>,
      std::allocator<BoardMapKey>, tsl::sh::power_of_two_growth_policy<2>,
      tsl::sh::exception_safety::basic, tsl::sh::sparsity::high>;

constexpr int kGroups = 5;

void SplitBoards(const std::filesystem::path&);

template <class Func> void ProcessBoards(int group, Func&& f) {
  constexpr size_t kBlock = 65536;
  auto fname = BoardPath(group);
  ClassReader<CompactBoard> reader(fname);
  while (true) {
    auto chunk = reader.ReadBatch(kBlock);
    for (auto& i : chunk) f(Board(i));
    if (chunk.size() < kBlock) break;
  }
}

BoardMap GetBoardMap(int group);

void BuildEdges();
