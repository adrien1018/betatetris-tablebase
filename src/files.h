#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>

// group = 0,1,2,3,4 (count/2%5)
std::filesystem::path BoardPath(int group);
std::filesystem::path BoardMapPath(int group);
std::filesystem::path EvaluateEdgePath(int group, int level);
std::filesystem::path PositionEdgePath(int group, int level);
std::filesystem::path ValuePath(int pieces);
std::filesystem::path ValueStatsPath(int pieces);
std::filesystem::path ProbPath(int pieces);
std::filesystem::path MoveIndexPath(int pieces);
std::filesystem::path MoveRangePath(int pieces_l, int pieces_r, int group);
std::filesystem::path MovePath(int group);
std::filesystem::path ThresholdOnePath(const std::string& name, int pieces);
std::filesystem::path ThresholdRangePath(const std::string& name, int pieces_l, int pieces_r, int group);
std::filesystem::path ThresholdPath(const std::string& name, int group);

std::vector<std::pair<int, int>> GetAvailableMoveRanges();
std::vector<std::pair<int, int>> GetAvailableThresholdRanges(const std::string& name);

std::filesystem::path SVDSamplePath(int group);
std::filesystem::path SVDSampleCountPath(int group);
std::filesystem::path SVDEvPath(int pieces);
std::filesystem::path SVDVarPath(int pieces);
std::filesystem::path SVDResultPath(bool ev);
std::filesystem::path SVDResultListPath(bool ev, int rank);

uint64_t BoardCount(const std::filesystem::path& board_file);
bool MkdirForFile(std::filesystem::path);
