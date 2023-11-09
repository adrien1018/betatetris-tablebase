#pragma once

#include <cstdint>
#include <filesystem>

extern std::filesystem::path kDataDir;

// group = 0,2,4,6,8 (count%10)
std::filesystem::path BoardPath(int group);
std::filesystem::path SearchEdgePath(int group);
std::filesystem::path ValuePath(int pieces);
std::filesystem::path ProbPath(int pieces);
uint64_t BoardCount(const std::filesystem::path& board_file);
