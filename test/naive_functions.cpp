#include <array>
#include <set>
#include "naive_functions.h"

namespace {

constexpr int kT = 7, kN = 20, kM = 10;

using Poly = std::array<std::pair<int, int>, 4>;
const std::vector<Poly> kBlocks[kT] = {
    {{{{1, 0}, {0, 0}, {0, 1}, {0, -1}}}, // T
     {{{1, 0}, {0, 0}, {-1, 0}, {0, -1}}},
     {{{0, -1}, {0, 0}, {0, 1}, {-1, 0}}},
     {{{1, 0}, {0, 0}, {0, 1}, {-1, 0}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, 1}}}, // J
     {{{-1, 0}, {0, 0}, {1, -1}, {1, 0}}},
     {{{-1, -1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {-1, 1}, {0, 0}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, 0}, {1, 1}}}, // Z
     {{{-1, 1}, {0, 0}, {0, 1}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, -1}, {1, 0}}}}, // O
    {{{{0, 0}, {0, 1}, {1, -1}, {1, 0}}}, // S
     {{{-1, 0}, {0, 0}, {0, 1}, {1, 1}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, -1}}}, // L
     {{{-1, -1}, {-1, 0}, {0, 0}, {1, 0}}},
     {{{-1, 1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {0, 0}, {1, 0}, {1, 1}}}},
    {{{{0, -2}, {0, -1}, {0, 0}, {0, 1}}}, // I
     {{{-2, 0}, {-1, 0}, {0, 0}, {1, 0}}}}};

constexpr int GetRow(int frame, Level level) {
  switch (level) {
    case kLevel18: return frame / 3;
    case kLevel19: return frame / 2;
    case kLevel29: return frame;
    case kLevel39: return frame * 2;
  }
  __builtin_unreachable();
}
constexpr bool IsDropFrame(int frame, Level level) {
  switch (level) {
    case kLevel18: return frame % 3 == 2;
    case kLevel19: return frame % 2 == 1;
    default: return true;
  }
}

inline std::pair<bool, Position> FreeDrop(const std::vector<ByteBoard>& b, int rot, int row, int col, Level level, int max_frame) {
  int max_row = GetRow(max_frame, level);
  while (row < max_row && row < 19 && b[rot][row+1][col]) row++;
  return {row >= max_row, {rot, row, col}};
}
inline Position FreeDrop(const std::vector<ByteBoard>& b, int rot, int row, int col) {
  while (row < 19 && b[rot][row+1][col]) row++;
  return {rot, row, col};
}
constexpr int kFinish = 1000;

void SimulateMove(
    const std::vector<ByteBoard>& b, Level level, const int taps[],
    int start_rot, int start_col, int start_frame, int end_frame, int num_lr, int num_ab, bool is_l, bool is_a,
    bool check_tuck, const std::set<Position>& non_tuck,
    std::vector<Position>& locked_placements, std::vector<std::pair<Position, int>>* continue_placements) {
  if (is_l ? (num_lr > start_col) : (num_lr > 9 - start_col)) return;
  if (is_a ? (num_ab > (int)b.size() / 2) : (num_ab > ((int)b.size() - 1) / 2)) return;
  if (is_l && num_lr == 0) return;
  if (is_a && num_ab == 0) return;
  int rot = start_rot, col = start_col, frame = start_frame;
  const int tot_taps = std::max(num_lr, num_ab);
  for (int tap = 0; tap < tot_taps; frame++) {
    int row = GetRow(frame, level);
    if (row >= 20 || !b[rot][row][col]) return;
    if (frame == taps[tap] + start_frame) {
      tap++;
      if (tap <= num_lr) {
        if (is_l) {
          if (!b[rot][row][--col]) return;
        } else {
          if (!b[rot][row][++col]) return;
        }
      }
      if (tap <= num_ab) {
        if (is_a) {
          if (!b[rot=(rot+1)%b.size()][row][col]) return;
        } else {
          if (!b[rot=(rot+b.size()-1)%b.size()][row][col]) return;
        }
      }
      if (tap == tot_taps) break;
    }
    if (IsDropFrame(frame, level)) {
      if (++row >= 20 || !b[rot][row][col]) return;
      if (level == kLevel39) {
        if (++row >= 20 || !b[rot][row][col]) return;
      }
    }
  }
  { // forward to tuck available
    auto pos = FreeDrop(b, rot, GetRow(frame, level), col, level, start_frame + taps[tot_taps]);
    if (!pos.first) {
      locked_placements.push_back(pos.second);
      return;
    }
    frame = start_frame + taps[tot_taps];
  }
  { // check continue
    auto pos = FreeDrop(b, rot, GetRow(frame, level), col, level, end_frame);
    if (pos.first) {
      continue_placements->push_back({pos.second, std::max(end_frame, frame)});
    } else if (end_frame == kFinish) {
      locked_placements.push_back(pos.second);
    }
  }
  if (!check_tuck) return;
  std::set<Position> tuck_placements;
  auto Insert = [&](const Position& p) {
    if (!non_tuck.count(p)) tuck_placements.insert(p);
  };
  for (; frame < end_frame; frame++) {
    int row = GetRow(frame, level);
    if (row >= 20 || !b[rot][row][col]) return;

    int nrow = GetRow(frame+1, level);
    int mrow = nrow - row == 2 ? nrow-1 : nrow;
    int arot = (rot+1)%b.size();
    int brot = (rot+b.size()-1)%b.size();
    if (col < 9 && b[rot][row][col+1]) {
      Insert(FreeDrop(b, rot, row, col+1));
      if (b.size() >= 2 && b[arot][row][col+1]) Insert(FreeDrop(b, arot, row, col+1));
      if (b.size() >= 4 && b[brot][row][col+1]) Insert(FreeDrop(b, brot, row, col+1));
    }
    if (col > 0 && b[rot][row][col-1]) {
      Insert(FreeDrop(b, rot, row, col-1));
      if (b.size() >= 2 && b[arot][row][col-1]) Insert(FreeDrop(b, arot, row, col-1));
      if (b.size() >= 4 && b[brot][row][col-1]) Insert(FreeDrop(b, brot, row, col-1));
    }
    if (b.size() >= 2 && b[arot][row][col]) {
      Insert(FreeDrop(b, arot, row, col));
      if (nrow < 20 && col < 9 && b[arot][mrow][col] && b[arot][nrow][col] && b[arot][nrow][col+1]) Insert(FreeDrop(b, arot, nrow, col+1));
      if (nrow < 20 && col > 0 && b[arot][mrow][col] && b[arot][nrow][col] && b[arot][nrow][col-1]) Insert(FreeDrop(b, arot, nrow, col-1));
    }
    if (b.size() >= 4 && b[brot][row][col]) {
      Insert(FreeDrop(b, brot, row, col));
      if (nrow < 20 && col < 9 && b[arot][mrow][col] && b[brot][nrow][col] && b[brot][nrow][col+1]) Insert(FreeDrop(b, brot, nrow, col+1));
      if (nrow < 20 && col > 0 && b[arot][mrow][col] && b[brot][nrow][col] && b[brot][nrow][col-1]) Insert(FreeDrop(b, brot, nrow, col-1));
    }

    if (IsDropFrame(frame, level)) {
      if (++row >= 20 || !b[rot][row][col]) return;
      if (level == kLevel39) {
        if (++row >= 20 || !b[rot][row][col]) return;
      }
    }
  }
  for (auto& i : tuck_placements) locked_placements.push_back(i);
}

void MoveSearch(
    const std::vector<ByteBoard>& b, Level level, const int taps[],
    int start_rot, int start_col, int start_frame, int end_frame,
    bool check_tuck, const std::set<Position>& non_tuck,
    std::vector<Position>& locked_placements, std::vector<std::pair<Position, int>>* continue_placements) {
  for (int lr = 0; lr <= 9; lr++) {
    for (int ab = 0; ab <= 2; ab++) {
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, false, false, check_tuck, non_tuck, locked_placements, continue_placements);
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, true, false, check_tuck, non_tuck, locked_placements, continue_placements);
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, false, true, check_tuck, non_tuck, locked_placements, continue_placements);
      SimulateMove(b, level, taps, start_rot, start_col, start_frame, end_frame, lr, ab, true, true, check_tuck, non_tuck, locked_placements, continue_placements);
    }
  }
}

} // namespace

std::vector<ByteBoard> GetPieceMap(const ByteBoard& field, int poly) {
  const size_t R = kBlocks[poly].size();
  std::vector<ByteBoard> ret(R, ByteBoard{});
  for (size_t r = 0; r < R; r++) {
    auto& pl = kBlocks[poly][r];
    for (int x = 0; x < kN; x++) {
      for (int y = 0; y < kM; y++) {
        bool flag = true;
        for (int i = 0; i < 4; i++) {
          int nx = pl[i].first + x, ny = pl[i].second + y;
          if (ny < 0 || nx >= kN || ny >= kM || (nx >= 0 && !field[nx][ny])) {
            flag = false;
            break;
          }
        }
        ret[r][x][y] = flag;
      }
    }
  }
  return ret;
}

ByteBoard PlacePiece(const ByteBoard& b, int poly, int r, int x, int y) {
  ByteBoard field(b);
  auto& pl = kBlocks[poly][r];
  for (auto& i : pl) {
    int nx = x + i.first, ny = y + i.second;
    if (nx >= kN || ny >= kM || nx < 0 || ny < 0) continue;
    field[nx][ny] = false;
  }
  return field;
}

int ClearLines(ByteBoard& field) {
  int i = kN - 1, j = kN - 1;
  for (; i >= 0; i--, j--) {
    bool flag = false;
    for (int y = 0; y < kM; y++) flag |= field[i][y];
    if (!flag) {
      j++;
    } else if (i != j) {
      field[j] = field[i];
    }
  }
  int ans = j + 1;
  for (; j >= 0; j--) std::fill(field[j].begin(), field[j].end(), true);
  return ans;
}

PossibleMoves NaiveGetPossibleMoves(const std::vector<ByteBoard>& b, Level level, int adj_frame, const int taps[]) {
  PossibleMoves ret;

  std::vector<Position> non_tuck;
  std::set<Position> non_tuck_set;
  MoveSearch(b, level, taps, 0, 5, 0, kFinish, false, non_tuck_set, non_tuck, nullptr);
  non_tuck_set = std::set<Position>(non_tuck.begin(), non_tuck.end());

  std::vector<std::pair<Position, int>> adj_starts;
  MoveSearch(b, level, taps, 0, 5, 0, adj_frame, true, non_tuck_set, ret.non_adj, &adj_starts);
  for (auto& i : adj_starts) {
    ret.adj.push_back({i.first, {}});
    MoveSearch(b, level, taps, i.first.r, i.first.y, i.second, kFinish, true, non_tuck_set, ret.adj.back().second, nullptr);
  }
  return ret;
}
