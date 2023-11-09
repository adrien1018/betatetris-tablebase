#include "files.h"
#include <string>

extern std::filesystem::path kDataDir;

std::filesystem::path BoardPath(int group) {
  return kDataDir / "boards" / std::to_string(group);
}
std::filesystem::path SearchEdgePath(int group) {
  return kDataDir / "edges" / (std::to_string(group) + ".search");
}
std::filesystem::path ValuePath(int pieces) {
  std::string r = std::to_string(value);
  return kDataDir / "values" / (std::string(4 - r.size(), '0') + r);
}
std::filesystem::path ProbPath(int pieces) {
  std::string r = std::to_string(value);
  return kDataDir / "probs" / (std::string(4 - r.size(), '0') + r);
}

uint64_t BoardCount(const std::filesystem::path& board_file) {
  return std::filesystem::file_size(board_file) / kBoardBytes;
}
