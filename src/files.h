#pragma once

#include <cstdint>
#include <filesystem>

// group = 0,1,2,3,4 (count/2%5)
std::filesystem::path BoardPath(int group);
std::filesystem::path EvaluateEdgePath(int group, int level);
std::filesystem::path PositionEdgePath(int group, int level);
std::filesystem::path ValuePath(int pieces);
std::filesystem::path ValueStatsPath(int pieces);
std::filesystem::path ProbPath(int pieces);
std::filesystem::path MovePath(int pieces);
std::filesystem::path MoveRangePath(int pieces_l, int pieces_r, int group);

std::filesystem::path SVDSamplePath(int group);
std::filesystem::path SVDSampleCountPath(int group);
std::filesystem::path SVDEvPath(int pieces);
std::filesystem::path SVDVarPath(int pieces);
std::filesystem::path SVDResultPath(bool ev);
std::filesystem::path SVDResultListPath(bool ev, int rank);

uint64_t BoardCount(const std::filesystem::path& board_file);
bool MkdirForFile(std::filesystem::path);
