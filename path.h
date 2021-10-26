#pragma once

#include <filesystem>

std::filesystem::path BoardPath(const std::filesystem::path& pdir, int group) {
  return pdir / (std::to_string(group) + ".board");
}
std::filesystem::path CountPath(const std::filesystem::path& pdir, int group) {
  return pdir / (std::to_string(group) + ".count");
}
std::filesystem::path EdgePath(const std::filesystem::path& pdir, int C, int group, const std::string& type) {
  return pdir / (std::to_string(group) + "." + std::to_string(C) + "." + type);
}
long BoardCount(const std::filesystem::path& board_file) {
  return std::filesystem::file_size(board_file) / kBoardBytes;
}
