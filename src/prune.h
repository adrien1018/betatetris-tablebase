#pragma once

#include <cstdint>
#include <array>
#include <vector>

#include "game.h"
#include "board.h"
#include "io_helpers.h"

using PruneMaskBase = std::array<std::vector<uint8_t>, kGroups>;

struct PruneMask : public PruneMaskBase {
  using PruneMaskBase::array;
  using PruneMaskBase::data;

  PruneMask(const uint8_t data[], size_t) {
    size_t ind = 0;
    for (auto& i : *this) ind += SimpleVecInput<4>(i, data + ind);
  }

  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 8;
  size_t NumBytes() const {
    size_t sz = 0;
    for (auto& i : *this) sz += i.size();
    return sz;
  }
  void GetBytes(uint8_t ret[]) const {
    size_t ind = 0;
    for (auto& i : *this) ind += SimpleVecOutput<4>(i, ret + ind);
  }
};

constexpr uint8_t kAllZeroValue = 0;
constexpr uint8_t kAllOneValue = (1 << kPieces) - 1;

PruneMask SameValueMask(uint8_t x);
PruneMask ReadMask(const std::string& path);
void ThresholdMask(
    PruneMask&& mask, int start_pieces, int end_pieces, float threshold, const std::string& path);
