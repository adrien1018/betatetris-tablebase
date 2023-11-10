#ifndef SEARCH_H_
#define SEARCH_H_

#include "../board.h"
#include "../position.h"

#include <bitset>
#include <vector>

template <int R>
using PositionList = std::vector<std::pair<Position, std::bitset<R * 200>>>;
using StateTuple = std::pair<int8_t, int8_t>; // nxt_lr, nxt_ab (in frames)

constexpr int kStartY = 5;

template <int R, int frames_per_drop, int frames_per_tap>
std::pair<std::vector<Position>, std::vector<std::pair<Position, StateTuple>>> SearchOne(
    const bool board[R][20][10], const std::pair<Position, StateTuple>& start,
    int num_frames) {
  auto Check = [&board](const Position& p) {
    return p.x >= 0 && p.x < 20 && p.y >= 0 && p.y < 10 && board[p.r][p.x][p.y];
  };
  if (!Check(start.first)) return {};

  std::bitset<200 * R> rx;
  StateTuple mem[2][R][10];
  auto current = mem[0];
  auto next = mem[1];
  int current_min = 0x7f;
  auto Decrement = [](const StateTuple& r) -> StateTuple {
    return {std::max(0, r.first - 1), std::max(0, r.second - 1)};
  };
  auto InsertCurrent = [&next,&current_min](int r, int y, const StateTuple& st) {
    if (st < next[r][y]) {
      next[r][y] = st;
      current_min = std::min((int)std::min(st.first, st.second), current_min);
    }
  };
  auto Clear = [](auto x) {
    memset(x, 0x7f, sizeof(mem[0]));
  };
  Clear(current);
  Clear(next);
  current[start.first.r][start.first.y] = start.second;

  int now_x = start.first.x;
  for (int i = 0; i < num_frames; i++) {
    const int to_drop = frames_per_drop - (i + 1) % frames_per_drop;
    current_min = 0x7f;
    auto Insert = [&,to_drop](int r, int y, const StateTuple& st) {
      if (to_drop == frames_per_drop) {
        if (Check({r, now_x + 1, y})) {
          InsertCurrent(r, y, st);
        } else {
          rx[r * 200 + now_x * 10 + y] = true;
        }
      } else {
        InsertCurrent(r, y, st);
      }
    };
    bool flag = false;
    for (int r = 0; r < R; r++) {
      for (int y = 0; y < 10; y++) {
        if (current[r][y].first == 0x7f) continue;
        flag = true;
        const auto val = current[r][y];
        Insert(r, y, Decrement(val));
        int ra = r == R - 1 ? 0 : r + 1;
        int rb = r == 0 ? R - 1 : r - 1;
        if (val.first == 0) {
          StateTuple nxt_state = Decrement({frames_per_tap, val.second});
          StateTuple both_state = {frames_per_tap - 1, frames_per_tap - 1};
          for (int ny : {y - 1, y + 1}) { // LR
            if (Check({r, now_x, ny})) {
              Insert(r, ny, nxt_state);
              if (R > 1 && val.second == 0) { // same-frame LR+AB
                if (Check({ra, now_x, ny})) Insert(ra, ny, both_state);
                if (Check({rb, now_x, ny})) Insert(rb, ny, both_state);
              }
            }
          }
        }
        if (R > 1 && val.second == 0) { // AB
          StateTuple nxt_state = Decrement({val.first, frames_per_tap});
          if (Check({ra, now_x, y})) Insert(ra, y, nxt_state);
          if (Check({rb, now_x, y})) Insert(rb, y, nxt_state);
        }
      }
    }
    if (!flag) break;
    // number of frames that no lock or input is possible
    int can_ignore = std::min(std::min(to_drop - 1, current_min), num_frames - i - 2);
    if (can_ignore > 0) {
      for (int r = 0; r < R; r++) {
        for (int y = 0; y < 10; y++) {
          if (next[r][y].first != 0x7f) {
            next[r][y].first -= can_ignore;
            next[r][y].second -= can_ignore;
          }
        }
      }
      i += can_ignore;
    }
    Clear(current);
    std::swap(current, next);
    if (to_drop == frames_per_drop) now_x++;
  }
  std::vector<Position> ret_vec;
  std::vector<std::pair<Position, StateTuple>> ret_mp;
  for (int r = 0; r < R; r++) {
    for (int y = 0; y < 10; y++) {
      if (current[r][y].first != 0x7f) ret_mp.push_back({{r, now_x, y}, current[r][y]});
    }
  }
  ret_vec.reserve(rx.count());
  for (int i = rx._Find_first(); i < (int)rx.size(); i = rx._Find_next(i)) {
    ret_vec.push_back({i / 200, i / 10 % 20, i % 10});
  }
  return {ret_vec, std::move(ret_mp)};
}


template <int R, int frames_per_drop, int frames_per_tap>
std::vector<std::bitset<R * 200>> SearchOneBatch(
    const bool board[R][20][10], const std::vector<std::pair<Position, StateTuple>>& start, int start_frame) {
  auto Check = [&board](const Position& p) {
    return p.x >= 0 && p.x < 20 && p.y >= 0 && p.y < 10 && board[p.r][p.x][p.y];
  };
  if (start.empty()) return {};

  const size_t N = start.size();
  if (N > 64) throw;
  std::vector<std::bitset<R * 200>> rx(N);
  // Each state is a bitset
  uint64_t mem[2][R][10][frames_per_tap][frames_per_tap] = {};
  auto current = mem[0];
  auto next = mem[1];
  auto Decrement = [](int x) -> int { return x - 1 + (x == 0); };
  auto InsertCurrent = [&next](int r, int y, int s1, int s2, uint64_t bs) { next[r][y][s1][s2] |= bs; };
  auto Clear = [](auto& x) { memset(x, 0, sizeof(mem[0])); };
  for (size_t n = 0; n < N; n++) {
    auto& cur = start[n];
    current[cur.first.r][cur.first.y][cur.second.first][cur.second.second] |= 1ul << n;
  }

  int now_x = start[0].first.x;
  for (int i = start_frame;; i++) {
    const int to_drop = frames_per_drop - (i + 1) % frames_per_drop;
    bool flag = false;
    for (int r = 0; r < R; r++) {
      for (int y = 0; y < 10; y++) {
        uint64_t cur_bs = 0;
        for (int s1 = 0; s1 < frames_per_tap; s1++) {
          for (int s2 = 0; s2 < frames_per_tap; s2++) {
            auto& bs = current[r][y][s1][s2];
            // update in (s1, s2) order so only the minimum is used to update
            bs &= ~cur_bs;
            if (bs == 0) continue;
            cur_bs |= bs;
            flag = true;

            // do not check board here
            int ds1 = Decrement(s1), ds2 = Decrement(s2), df = frames_per_tap - 1;
            InsertCurrent(r, y, ds1, ds2, bs);
            int ra = r == R - 1 ? 0 : r + 1;
            int rb = r == 0 ? R - 1 : r - 1;
            if (s1 == 0) {
              for (int ny : {y - 1, y + 1}) { // LR
                // board check is necessary for same-frame; move it here for simplicity
                if (!Check({r, now_x, ny})) continue;
                InsertCurrent(r, ny, df, ds2, bs);
                if (R > 1 && s2 == 0) { // same-frame LR+AB
                  InsertCurrent(ra, ny, df, df, bs);
                  InsertCurrent(rb, ny, df, df, bs);
                }
              }
            }
            if (R > 1 && s2 == 0) { // AB
              InsertCurrent(ra, y, ds1, df, bs);
              InsertCurrent(rb, y, ds1, df, bs);
            }
          }
        }
      }
    }
    if (!flag) break;
    for (int r = 0; r < R; r++) { // check board altogether
      for (int y = 0; y < 10; y++) {
        if (!Check({r, now_x, y})) {
          memset(next[r][y], 0, sizeof(next[r][y]));
        } else if (to_drop == frames_per_drop && !Check({r, now_x + 1, y})) { // lock
          uint64_t bs = 0;
          for (int s1 = 0; s1 < frames_per_tap; s1++) {
            for (int s2 = 0; s2 < frames_per_tap; s2++) bs |= next[r][y][s1][s2];
          }
          while (bs) {
            int n = __builtin_ctzll(bs);
            rx[n][r * 200 + now_x * 10 + y] = true;
            bs &= ~(1ll << n);
          }
          memset(next[r][y], 0, sizeof(next[r][y]));
        }
      }
    }
    Clear(current);
    std::swap(current, next);
    if (to_drop == frames_per_drop) now_x++;
  }
  return rx;
}

template <int R, int frames_per_drop, int frames_per_tap, int microadj_delay>
PositionList<R> SearchMoves(const std::array<Board, R>& mp) {
  bool board[R][20][10];
  for (int r = 0; r < R; r++) {
    for (int y = 0; y < 10; y++) {
      uint32_t col = mp[r].Column(y);
      for (int x = 0; x < 20; x++) board[r][x][y] = col >> x & 1;
    }
  }
  if (!board[0][0][kStartY]) return {};

  auto before_adj = SearchOne<R, frames_per_drop, frames_per_tap>(board, {{0, 0, kStartY}, {0, 0}}, microadj_delay);
  PositionList<R> ret;
  ret.reserve(before_adj.first.size() + before_adj.second.size());
  for (auto& pos : before_adj.first) {
    std::bitset<R * 200> d;
    d[pos.r * 200 + pos.x * 10 + pos.y] = true;
    ret.push_back({pos, d});
  }

  auto dd = SearchOneBatch<R, frames_per_drop, frames_per_tap>(board, before_adj.second, microadj_delay);
  size_t i = 0;
  for (auto& st : before_adj.second) {
    ret.push_back({st.first, std::move(dd[i])});
    i += 1;
  }
  return ret;
}

#endif // SEARCH_H_
