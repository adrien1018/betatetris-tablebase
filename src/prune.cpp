#include "prune.h"

#include <spdlog/spdlog.h>

#include "evaluate.h"
#include "board_set.h"

namespace {

void UpdateMask(std::vector<uint8_t>& mask, const std::vector<MoveEval>& values, float threshold) {
  if (mask.size() != values.size()) throw std::length_error("incorrect mask size");
  for (size_t i = 0; i < mask.size(); i++) {
    __m256 cmp = _mm256_cmp_ps(_mm256_set1_ps(threshold), values[i].ev_vec, _CMP_LE_OQ);
    mask[i] |= pext<uint32_t>(_mm256_movemask_epi8(_mm256_castps_si256(cmp)), 0x11111111) & kAllOneValue;
  }
}

} // namespace

PruneMask SameValueMask(uint8_t x) {
  std::vector<size_t> offsets[kGroups];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  PruneMask ret;
  for (int i = 0; i < kGroups; i++) ret[i].resize(offsets[i].back(), x);
  return ret;
}

PruneMask ReadMask(const std::string& path) {
  CompressedClassReader<PruneMask> reader(path);
  return reader.ReadOne();
}

void WriteMask(const PruneMask& mask, const std::string& path) {
  CompressedClassWriter<PruneMask> writer(path, 1024, -2);
  writer.Write(mask);
}

void ThresholdMask(
    PruneMask&& mask, int start_pieces, int end_pieces, float threshold, const std::string& path) {
  std::vector<size_t> offsets[kGroups];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  spdlog::info("Reading values from piece {}", start_pieces);
  int start_group = GetGroupByPieces(start_pieces);
  std::vector<MoveEval> values = ReadValuesEvOnly(start_pieces, offsets[start_group].back());
  spdlog::info("Update mask from piece {}", start_pieces);
  UpdateMask(mask[start_group], values, threshold);
  for (int pieces = start_pieces - 1; pieces >= end_pieces; pieces--) {
    spdlog::info("Update mask from piece {}", pieces);
    int group = GetGroupByPieces(pieces);
    values = CalculatePiece(pieces, values, offsets[group]);
    UpdateMask(mask[group], values, threshold);
  }
  spdlog::info("Writing mask");
  WriteMask(mask, path);
}
