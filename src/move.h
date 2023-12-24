#pragma once

#include <vector>
#include <stdexcept>
#include "game.h"
#include "board.h"
#include "position.h"
#include "io_helpers.h"

// store range in uint8_t
static_assert(kLineCap <= 255 * 2);

using NodeMoveIndex = SimpleIOArray<uint8_t, 7>;

struct MoveIndexRange {
  uint8_t start, end;
  NodeMoveIndex idx;
  constexpr auto operator<=>(const MoveIndexRange&) const = default;
};

struct MovePositionRange {
  uint8_t start, end;
  std::array<Position, 7> pos;
  constexpr auto operator<=>(const MovePositionRange&) const = default;
};

struct NodeMoveIndexRange {
  std::vector<MoveIndexRange> ranges;

  NodeMoveIndexRange() {}
  template <class Iter>
  NodeMoveIndexRange(Iter begin, Iter end, uint8_t start) {
    size_t sz = std::distance(begin, end);
    if (sz == 0) return;
    if (sz + start >= 256) throw std::length_error("range too large");
    uint8_t prev = 0;
    Iter prev_it = begin++;
    for (uint8_t i = 1; i < sz; i++, ++begin) {
      if (*begin != *prev_it) {
        ranges.push_back({uint8_t(start + prev), uint8_t(start + i), *prev_it});
        prev = i;
        prev_it = begin;
      }
    }
    ranges.push_back({uint8_t(start + prev), uint8_t(start + sz), *prev_it});
  }

  NodeMoveIndexRange(const uint8_t buf[], size_t sz) {
    if (sz % sizeof(MoveIndexRange) != 0) throw std::length_error("invalid size");
    ranges.resize(sz / sizeof(MoveIndexRange));
    memcpy(ranges.data(), buf, sz);
  }

  NodeMoveIndexRange& operator<<=(const NodeMoveIndexRange& x) {
    size_t v = 0;
    if (ranges.size() && x.ranges.size()) {
      if (ranges.back().end != x.ranges[0].start) throw std::logic_error("not subsequent: " + std::to_string(ranges.back().end) + "," + std::to_string(x.ranges[0].start));
      if (ranges.back().idx == x.ranges[0].idx) {
        ranges.back().end = x.ranges[0].end;
        v++;
      }
    }
    for (; v < x.ranges.size(); v++) ranges.push_back(x.ranges[v]);
    return *this;
  }

  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;
  size_t NumBytes() const {
    return sizeof(MoveIndexRange) * ranges.size();
  }

  void GetBytes(uint8_t ret[]) const {
    memcpy(ret, ranges.data(), NumBytes());
  }
};

struct NodeMovePositionRange {
  static constexpr int kElementSize = 16;

  std::vector<MovePositionRange> ranges;

  NodeMovePositionRange() {}
  NodeMovePositionRange(const uint8_t buf[], size_t sz) {
    if (sz % kElementSize != 0) throw std::length_error("invalid size");
    ranges.resize(sz / kElementSize);
    for (size_t i = 0; i < ranges.size(); i++) {
      ranges[i].start = buf[i*kElementSize];
      ranges[i].end = buf[i*kElementSize+1];
      for (size_t j = 0; j < kPieces; j++) {
        new(&ranges[i].pos[j]) Position(buf + (i*kElementSize + 2 + j*2), 2);
      }
    }
  }

  NodeMovePositionRange& operator<<=(const MovePositionRange& x) {
    if (ranges.empty()) {
      ranges.push_back(x);
    } else if (ranges.back().end != x.start) {
      throw std::logic_error("not subsequent: " + std::to_string(ranges.back().end) + "," + std::to_string(x.start));
    } else if (ranges.back().pos != x.pos) {
      ranges.push_back(x);
    } else {
      ranges.back().end = x.end;
    }
    return *this;
  }

  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;
  size_t NumBytes() const {
    return kElementSize * ranges.size();
  }

  void GetBytes(uint8_t ret[]) const {
    for (size_t i = 0; i < ranges.size(); i++) {
      ret[i*kElementSize] = ranges[i].start;
      ret[i*kElementSize+1] = ranges[i].end;
      for (size_t j = 0; j < kPieces; j++) {
        ranges[i].pos[j].GetBytes(ret + (i*kElementSize + 2 + j*2));
      }
    }
  }
};

void RunCalculateMoves(int start_pieces, int end_pieces);
void MergeRanges(int pieces_l, int pieces_r, bool delete_after);
void MergeFullRanges();
