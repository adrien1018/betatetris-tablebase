#include "files.h"

#include <regex>
#include <string>
#include <algorithm>
#include <filesystem>
#include "board.h"
#include "config.h"

namespace fs = std::filesystem;

namespace {

std::string NumToStr(int num, size_t lpad = 4) {
  std::string r = std::to_string(num);
  return std::string(lpad - r.size(), '0') + r;
}

std::vector<std::pair<int, int>> GetAvailableRanges(const fs::path path) {
  std::regex pattern("[0-4].([0-9]+)-([0-9]+)");
  std::vector<std::pair<int, int>> ret;
  for (auto const& dir_entry : std::filesystem::directory_iterator{path}) {
    std::cmatch match;
    if (std::regex_match(dir_entry.path().filename().c_str(), match, pattern)) {
      ret.push_back({std::stoi(match[1]), std::stoi(match[2])});
    }
  }
  std::sort(ret.begin(), ret.end());
  ret.resize(std::unique(ret.begin(), ret.end()) - ret.begin());
  return ret;
}

} // namespace

fs::path BoardPath(int group) {
  return kDataDir / "boards" / std::to_string(group);
}
fs::path BoardMapPath(int group) {
  return kDataDir / "boards" / (std::to_string(group) + ".map");
}
fs::path EvaluateEdgePath(int group, int level) {
  return kDataDir / "edges" / (std::to_string(group) + ".l" + std::to_string(level) + ".eval");
}
fs::path PositionEdgePath(int group, int level) {
  return kDataDir / "edges" / (std::to_string(group) + ".l" + std::to_string(level) + ".pos");
}
fs::path ValuePath(int pieces) {
  return kDataDir / "values" / NumToStr(pieces);
}
fs::path ValueStatsPath(int pieces) {
  return kDataDir / "value_stats" / NumToStr(pieces);
}
fs::path ProbPath(int pieces) {
  return kDataDir / "probs" / NumToStr(pieces);
}
fs::path MoveIndexPath(int pieces) {
  return kDataDir / "moves" / NumToStr(pieces);
}
fs::path MoveRangePath(int pieces_l, int pieces_r, int group) {
  return kDataDir / "moves" / (std::to_string(group) + '.' + NumToStr(pieces_l) + '-' + NumToStr(pieces_r));
}
fs::path MovePath(int group) {
  return kDataDir / "moves" / (std::to_string(group) + ".all");
}
fs::path ThresholdOnePath(const std::string& name, int pieces) {
  return kDataDir / "threshold" / name / NumToStr(pieces);
}
fs::path ThresholdRangePath(const std::string& name, int pieces_l, int pieces_r, int group) {
  return kDataDir / "threshold" / name / (std::to_string(group) + '.' + NumToStr(pieces_l) + '-' + NumToStr(pieces_r));
}
fs::path ThresholdPath(const std::string& name, int group) {
  return kDataDir / "threshold" / name / (std::to_string(group) + ".all");
}

std::vector<std::pair<int, int>> GetAvailableMoveRanges() {
  return GetAvailableRanges(kDataDir / "moves");
}
std::vector<std::pair<int, int>> GetAvailableThresholdRanges(const std::string& name) {
  return GetAvailableRanges(kDataDir / "threshold" / name);
}

fs::path SVDSamplePath(int group) {
  return kDataDir / "svd" / (std::to_string(group) + ".sample");
}
fs::path SVDSampleCountPath(int group) {
  return kDataDir / "svd" / (std::to_string(group) + ".count");
}
fs::path SVDEvPath(int pieces) {
  return kDataDir / "svd" / (NumToStr(pieces) + ".ev");
}
fs::path SVDVarPath(int pieces) {
  return kDataDir / "svd" / (NumToStr(pieces) + ".var");
}
fs::path SVDResultPath(bool ev) {
  return kDataDir / "svd" / "result" / (std::string("overall") + (ev ? "-ev" : "-var") + ".txt");
}
fs::path SVDResultListPath(bool ev, int rank) {
  return kDataDir / "svd" / "result" / (NumToStr(rank, 3) + (ev ? "-ev" : "-var") + ".txt");
}

uint64_t BoardCount(const fs::path& board_file) {
  return fs::file_size(board_file) / kBoardBytes;
}

bool MkdirForFile(fs::path path) {
  auto x = path.remove_filename();
  if (x.empty()) return true;
  return fs::create_directories(path.remove_filename());
}
