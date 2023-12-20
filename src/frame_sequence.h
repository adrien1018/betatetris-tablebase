#pragma once

#include "move_search.h"

struct FrameInput {
  static const FrameInput A;
  static const FrameInput B;
  static const FrameInput L;
  static const FrameInput R;
  uint8_t value;
  FrameInput operator|=(FrameInput a) {
    value |= a.value;
    return *this;
  }
  bool IsA() const { return value & 1; }
  bool IsB() const { return value & 2; }
  bool IsL() const { return value & 4; }
  bool IsR() const { return value & 8; }
};

inline constexpr FrameInput FrameInput::A = FrameInput{1};
inline constexpr FrameInput FrameInput::B = FrameInput{2};
inline constexpr FrameInput FrameInput::L = FrameInput{4};
inline constexpr FrameInput FrameInput::R = FrameInput{8};

inline FrameInput operator|(FrameInput a, FrameInput b) {
  return a |= b;
}

using FrameSequence = std::vector<FrameInput>;

namespace move_search {

// return a frame range
template <Level level, int R, class Taps>
constexpr std::pair<int, int> GetFrameRange(
    const std::array<Board, R>& board, bool is_tuck,
    int initial_rot, int initial_col, int initial_frame, int target_rot, int target_col) {
  constexpr Taps taps{};
  int num_lr_tap = abs(target_col - initial_col);
  int num_ab_tap = (target_rot + 4 - initial_rot) % R;
  bool is_l = target_col < initial_col;
  bool is_a = true;
  if (num_ab_tap == 3) num_ab_tap = 1, is_a = false;
  int num_taps = std::max(num_lr_tap, num_ab_tap);

  Column target_column = board[target_rot].Column(target_col);
  int target_frame = initial_frame + (is_tuck ? taps[num_taps] : num_taps ? taps[num_taps - 1] : 0);
  int prev_row = GetRow(initial_frame, level);
  int cur_rot = initial_rot, cur_col = initial_col;
  if (!board[cur_rot].IsCellSet(prev_row, cur_col)) return {-1, -1};
  for (int i = 0; i < num_taps; i++) {
    int cur_row = GetRow(initial_frame + taps[i], level);
    if (!board[cur_rot].IsColumnRangeSet(prev_row, cur_row + 1, cur_col)) return {-1, -1};
    if (i < num_lr_tap) {
      is_l ? cur_col-- : cur_col++;
      if (!board[cur_rot].IsCellSet(cur_row, cur_col)) return {-1, -1};
    }
    if (i < num_ab_tap) {
      cur_rot = (cur_rot + (is_a ? 1 : R - 1)) % R;
      if (!board[cur_rot].IsCellSet(cur_row, cur_col)) return {-1, -1};
    }
    prev_row = cur_row;
  }
  if (is_tuck) {
    int cur_row = GetRow(initial_frame + taps[num_taps], level);
    if (!board[cur_rot].IsColumnRangeSet(prev_row, cur_row + 1, cur_col)) return {-1, -1};
    prev_row = cur_row;
  }
  int final_row = 31 - clz((target_column + (1 << prev_row)) ^ target_column) - 1;
  return {target_frame, GetLastFrameOnRow(final_row, level)};
}

template <int R, class Taps>
void GenerateSequence(
    FrameSequence& seq, int initial_rot, int initial_col, int initial_frame,
    int target_rot, int target_col, size_t min_frames) {
  constexpr Taps taps{};
  int num_lr_tap = abs(target_col - initial_col);
  int num_ab_tap = (target_rot + R - initial_rot) % R;
  bool is_l = target_col < initial_col;
  bool is_a = true;
  if (num_ab_tap == 3) num_ab_tap = 1, is_a = false;
  int num_taps = std::max(num_lr_tap, num_ab_tap);
  seq.resize(initial_frame, FrameInput{});
  for (int i = 0; i < num_taps; i++) {
    seq.resize(initial_frame + taps[i], FrameInput{});
    FrameInput cur{};
    if (i < num_lr_tap) cur |= is_l ? FrameInput::L : FrameInput::R;
    if (i < num_ab_tap) cur |= is_a ? FrameInput::A : FrameInput::B;
    seq.push_back(cur);
  }
  seq.resize(initial_frame + taps[num_taps], FrameInput{});
  if (min_frames > seq.size()) seq.resize(min_frames, FrameInput{});
}

// should only be used for reachable positions; otherwise the result would probably be incorrect
template <Level level, int R, class Taps>
NOINLINE void CalculateSequence(
    const std::array<Board, R>& board, FrameSequence& seq,
    int initial_rot, int initial_col, int initial_frame,
    const Position& target, size_t min_frames) {
#ifndef _MSC_VER
  static_assert(IsTapTable<Taps>::value);
#endif
  int max_height = 0;
  for (auto& i : board) max_height = std::max(max_height, i.Height());

  Column target_column = board[target.r].Column(target.y);
  int first_reachable_row = 31 - clz<uint32_t>(~(target_column << 1 | -(2 << target.x)));
  int first_reachable_frame = GetFirstFrameOnRow(first_reachable_row, level);
  int last_reachable_frame = GetLastFrameOnRow(target.x, level);
  {
    auto [frame_start, frame_end] = GetFrameRange<level, R, Taps>(
        board, false, initial_rot, initial_col, initial_frame, target.r, target.y);
    if (frame_end >= first_reachable_frame && last_reachable_frame >= frame_start) {
      GenerateSequence<R, Taps>(seq, initial_rot, initial_col, initial_frame, target.r, target.y, min_frames);
      return;
    }
  }

  constexpr TuckTypeTable<R> tucks;
  Frames target_frames = (2ll << GetLastFrameOnRow(target.x, level)) - (1ll << first_reachable_frame);
  for (int i = 0; i < TuckTypes(R); i++) {
    auto& tuck = tucks.table[i];
    int intermediate_rot = (target.r + R - tuck.delta_rot) % R;
    int intermediate_col = target.y - tuck.delta_col;
    if (intermediate_col >= 10 || intermediate_col < 0) continue;
    auto [frame_start, frame_end] = GetFrameRange<level, R, Taps>(
        board, true, initial_rot, initial_col, initial_frame, intermediate_rot, intermediate_col);
    if (frame_start == -1) continue;
    Frames frame_mask_1 = (2ll << frame_end) - (1ll << frame_start);
    frame_mask_1 &= target_frames >> tuck.delta_frame;
    Frames frame_mask_2 = 0; // for tuck-spin on different frame
    switch (i) {
      case 0: case 1: case 2: case 7: break; // no intermediate
      case 3: case 4: case 8: case 9: {
        frame_mask_1 &= ColumnToNormalFrameMask<level>(board[intermediate_rot].Column(target.y));
        break;
      }
      default: { // 5, 6, 10, 11
        frame_mask_2 = frame_mask_1 & ColumnToDropFrameMask<level>(board[target.r].Column(intermediate_col));
        frame_mask_1 &= ColumnToDropFrameMask<level>(board[intermediate_rot].Column(target.y));
        break;
      }
    }
    if (frame_mask_1) {
      int tuck_frame = ctz(frame_mask_1);
      GenerateSequence<R, Taps>(seq, initial_rot, initial_col, initial_frame, intermediate_rot, intermediate_col, tuck_frame);
      switch (i) {
        case 0: seq.push_back(FrameInput::L); break;
        case 1: seq.push_back(FrameInput::R); break;
        case 2: seq.push_back(FrameInput::A); break;
        case 3: seq.push_back(FrameInput::L | FrameInput::A); break;
        case 4: seq.push_back(FrameInput::R | FrameInput::A); break;
        case 5: seq.push_back(FrameInput::L); seq.push_back(FrameInput::A); break;
        case 6: seq.push_back(FrameInput::R); seq.push_back(FrameInput::A); break;
        case 7: seq.push_back(FrameInput::B); break;
        case 8: seq.push_back(FrameInput::L | FrameInput::B); break;
        case 9: seq.push_back(FrameInput::R | FrameInput::B); break;
        case 10: seq.push_back(FrameInput::L); seq.push_back(FrameInput::B); break;
        case 11: seq.push_back(FrameInput::R); seq.push_back(FrameInput::B); break;
      }
      if (min_frames > seq.size()) seq.resize(min_frames, FrameInput{});
      return;
    }
    if (frame_mask_2) {
      int tuck_frame = ctz(frame_mask_2);
      GenerateSequence<R, Taps>(seq, initial_rot, initial_col, initial_frame, intermediate_rot, intermediate_col, tuck_frame);
      switch (i) {
        case 5: seq.push_back(FrameInput::A); seq.push_back(FrameInput::L); break;
        case 6: seq.push_back(FrameInput::A); seq.push_back(FrameInput::R); break;
        case 10: seq.push_back(FrameInput::B); seq.push_back(FrameInput::L); break;
        case 11: seq.push_back(FrameInput::B); seq.push_back(FrameInput::R); break;
      }
      if (min_frames > seq.size()) seq.resize(min_frames, FrameInput{});
      return;
    }
  }
}

} // namespace move_search

template <Level level, int R, class Taps>
FrameSequence GetFrameSequence(
    const std::array<Board, R>& board,
    int initial_rot, int initial_col, int initial_frame,
    const Position& target, size_t min_frames = 0) {
  FrameSequence seq;
  move_search::CalculateSequence<level, R, Taps>(board, seq, initial_rot, initial_col, initial_frame, target, min_frames);
  return seq;
}

template <Level level, class Taps>
FrameSequence GetFrameSequenceStart(const Board& b, int piece, int adj_delay, const Position& target) {
#define PIECE_CASE_TMPL_ARGS ,Taps
#define PIECE_CASE_ARGS ,0,Position::Start.y,0,target,adj_delay
  DO_PIECE_CASE(GetFrameSequence, b);
#undef PIECE_CASE_TMPL_ARGS
#undef PIECE_CASE_ARGS
}

template <Level level, class Taps>
void GetFrameSequenceAdj(
    FrameSequence& seq, const Board& b, int piece, const Position& premove,
    const Position& target) {
#define PIECE_CASE_TMPL_ARGS ,Taps
#define PIECE_CASE_ARGS ,seq,premove.r,premove.y,seq.size(),target,0
  DO_PIECE_CASE(move_search::CalculateSequence, b);
#undef PIECE_CASE_TMPL_ARGS
#undef PIECE_CASE_ARGS
}

template <Level level, int R>
std::pair<Position, bool> SimulateMove(const std::array<Board, R>& board, const FrameSequence& seq, bool until_lock) {
  Position pos = Position::Start;
  int charge = 0;
  FrameInput prev_input{};
  for (size_t frame = 0; frame < seq.size(); frame++) {
    auto& input = seq[frame];
    if (input.IsL()) {
      bool is_available = pos.y > 0 && board[pos.r].IsCellSet(pos.x, pos.y - 1);
      if (!prev_input.IsL()) {
        charge = 0;
        if (is_available) pos.y--;
      } else if (!is_available) {
        // nothing; prevent charge increase
      } else if (++charge == 6) {
        charge = 0;
        pos.y--;
      }
    } else if (input.IsR()) {
      bool is_available = pos.y < 9 && board[pos.r].IsCellSet(pos.x, pos.y + 1);
      if (!prev_input.IsR()) {
        charge = 0;
        if (is_available) pos.y++;
      } else if (!is_available) {
        // nothing; prevent charge increase
      } else if (++charge == 6) {
        charge = 0;
        pos.y++;
      }
    }
    if (input.IsA() && !prev_input.IsA()) {
      int new_r = (pos.r + 1) % R;
      if (board[new_r].IsCellSet(pos.x, pos.y)) pos.r = new_r;
    } else if (input.IsB() && !prev_input.IsB()) {
      int new_r = (pos.r + R - 1) % R;
      if (board[new_r].IsCellSet(pos.x, pos.y)) pos.r = new_r;
    }
    for (int i = 0; i < move_search::NumDrops(frame, level); i++) {
      if (pos.x == 19 || !board[pos.r].IsCellSet(pos.x + 1, pos.y)) return {pos, true};
      pos.x++;
    }
    prev_input = input;
  }
  if (!until_lock) return {pos, false};
  while (true) {
    if (pos.x == 19 || !board[pos.r].IsCellSet(pos.x + 1, pos.y)) return {pos, true};
    pos.x++;
  }
}

template <Level level>
std::pair<Position, bool> SimulateMove(const Board& b, int piece, const FrameSequence& seq, bool until_lock) {
#define PIECE_CASE_TMPL_ARGS
#define PIECE_CASE_ARGS ,seq,until_lock
  DO_PIECE_CASE(SimulateMove, b);
#undef PIECE_CASE_TMPL_ARGS
#undef PIECE_CASE_ARGS
}
