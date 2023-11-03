#pragma once

#include <filesystem>

#include "board.h"

std::filesystem::path BoardPath(const std::filesystem::path& pdir, int group) {
  return pdir / (std::to_string(group) + ".board");
}
std::filesystem::path CountPath(const std::filesystem::path& pdir, int group) {
  return pdir / (std::to_string(group) + ".count");
}
std::filesystem::path EdgePath(const std::filesystem::path& pdir, int C, int group, const std::string& type) {
  return pdir / (std::to_string(group) + "." + std::to_string(C) + "." + type);
}
std::filesystem::path ValuePath(const std::filesystem::path& pdir, int value) {
  std::string r = std::to_string(value);
  return pdir / ("values." + std::string(3 - r.size(), '0') + r);
}
long BoardCount(const std::filesystem::path& board_file) {
  return std::filesystem::file_size(board_file) / kBoardBytes;
}
