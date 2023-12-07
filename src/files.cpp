#include "files.h"

#include <string>
#include <filesystem>
#include "board.h"
#include "config.h"

namespace fs = std::filesystem;

namespace {

std::string NumToStr(int num, size_t lpad = 4) {
  std::string r = std::to_string(num);
  return std::string(lpad - r.size(), '0') + r;
}

} // namespace

fs::path BoardPath(int group) {
  return kDataDir / "boards" / std::to_string(group);
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
  return fs::create_directories(path.remove_filename());
}
