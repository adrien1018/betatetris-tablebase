#pragma once

#include <cstdint>
#include <filesystem>

// group = 0,1,2,3,4 (count/2%5)
std::filesystem::path BoardPath(int group);
std::filesystem::path SearchEdgePath(int group);
std::filesystem::path ValuePath(int pieces);
std::filesystem::path ProbPath(int pieces);
uint64_t BoardCount(const std::filesystem::path& board_file);
bool MkdirForFile(std::filesystem::path);
