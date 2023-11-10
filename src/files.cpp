#include "files.h"

#include <string>
#include <filesystem>
#include "board.h"
#include "config.h"

namespace fs = std::filesystem;

fs::path BoardPath(int group) {
  return kDataDir / "boards" / std::to_string(group);
}
fs::path SearchEdgePath(int group) {
  return kDataDir / "edges" / (std::to_string(group) + ".search");
}
fs::path ValuePath(int pieces) {
  std::string r = std::to_string(pieces);
  return kDataDir / "values" / (std::string(4 - r.size(), '0') + r);
}
fs::path ProbPath(int pieces) {
  std::string r = std::to_string(pieces);
  return kDataDir / "probs" / (std::string(4 - r.size(), '0') + r);
}

uint64_t BoardCount(const fs::path& board_file) {
  return fs::file_size(board_file) / kBoardBytes;
}

bool MkdirForFile(fs::path path) {
  return fs::create_directories(path.remove_filename());
}
