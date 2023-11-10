#pragma once

#include <vector>
#include <filesystem>
#include <tsl/sparse_map.h>
#include "board.h"

using BoardMapKey = std::pair<Board, uint64_t>;
using BoardMap = tsl::sparse_map<Board, uint64_t, std::hash<Board>, std::equal_to<Board>,
      std::allocator<BoardMapKey>, tsl::sh::power_of_two_growth_policy<2>,
      tsl::sh::exception_safety::basic, tsl::sh::sparsity::high>;

void SplitBoards(const std::filesystem::path&);

BoardMap GetBoardMap(int group);
std::vector<Board> GetBoards(int group);

void BuildEdges();
