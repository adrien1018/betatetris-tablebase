#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <array>
#include <tuple>
#include <bitset>
#include <vector>

#include <immintrin.h>

class alignas(32) Board;
constexpr Board operator|(const Board& x, const Board& y);
constexpr Board operator&(const Board& x, const Board& y);

// A 20x10 board is represented by 4 64-bit integers.
// Each integer represents 3 columns except b4. b1 is the leftmost 3 columns.
//   column 0 (leftmost): bit 0(topmost)-19(bottommost)
//   column 1: bit 22-41
//   column 2: bit 44-63
// A bit of 1 means an empty cell; 0 otherwise.
class alignas(32) Board {
 private:
  // 1 wide, offset = (2, 0)
  static constexpr uint64_t kIPiece1_ = 0xf;
  // 2 wide, offset = (0, 1)
  static constexpr uint64_t kOPiece_ = 0xc00003;
  // 2 wide, offset = (1, 0)
  static constexpr uint64_t kTPiece3_ = 0x800007;
  static constexpr uint64_t kJPiece3_ = 0x400007;
  static constexpr uint64_t kZPiece1_ = 0xc00006;
  static constexpr uint64_t kSPiece1_ = 0x1800003;
  static constexpr uint64_t kLPiece3_ = 0x1000007;
  // 2 wide, offset = (1, 1)
  static constexpr uint64_t kTPiece1_ = 0x1c00002;
  static constexpr uint64_t kJPiece1_ = 0x1c00004;
  static constexpr uint64_t kLPiece1_ = 0x1c00001;
  // 3 wide, offset = (0, 1)
  static constexpr uint64_t kTPiece0_ = 0x100000c00001;
  static constexpr uint64_t kJPiece0_ = 0x300000400001;
  static constexpr uint64_t kZPiece0_ = 0x200000c00001;
  static constexpr uint64_t kSPiece0_ = 0x100000c00002;
  static constexpr uint64_t kLPiece0_ = 0x100000400003;
  // 3 wide, offset = (1, 1)
  static constexpr uint64_t kTPiece2_ = 0x200000c00002;
  static constexpr uint64_t kJPiece2_ = 0x200000800003;
  static constexpr uint64_t kLPiece2_ = 0x300000800002;
  // 4 wide, offset = (0, 2)
  static constexpr uint64_t kIPiece0a_ = 0x100000400001;
  static constexpr uint64_t kIPiece0b_ = 0x400001;
  static constexpr uint64_t kIPiece0c_ = 0x1;

  constexpr Board Place1Wide_(uint64_t piece, int x, int y, int ox) const {
    Board r = *this;
    x -= ox;
    switch (y) {
      case 0: case 1: case 2: r.b1 &= ~(piece << (x + y * 22)); break;
      case 3: case 4: case 5: r.b2 &= ~(piece << (x + (y - 3) * 22)); break;
      case 6: case 7: case 8: r.b3 &= ~(piece << (x + (y - 6) * 22)); break;
      case 9: r.b4 &= ~(piece << x); break;
      default: __builtin_unreachable();
    }
    return r;
  }
  constexpr Board Place2Wide_(uint64_t piece, int x, int y, int ox, int oy) const {
    Board r = *this;
    x -= ox;
    y -= oy;
    switch (y) {
      case 2: r.b2 &= ~(piece >> (22 - x)); // fallthrough
      case 0: case 1: r.b1 &= ~(piece << (x + y * 22)); break;
      case 5: r.b3 &= ~(piece >> (22 - x)); // fallthrough
      case 3: case 4: r.b2 &= ~(piece << (x + (y - 3) * 22)); break;
      case 8: r.b4 &= ~(piece >> (22 - x)); // fallthrough
      case 6: case 7: r.b3 &= ~(piece << (x + (y - 6) * 22)); break;
      default: __builtin_unreachable();
    }
    return r;
  }
  constexpr Board Place3Wide_(uint64_t piece, int x, int y, int ox, int oy) const {
    Board r = *this;
    x -= ox;
    y -= oy;
    switch (y) {
      case 1: case 2: r.b2 &= ~(piece >> (66 - x - y * 22)); // fallthrough
      case 0: r.b1 &= ~(piece << (x + y * 22)); break;
      case 4: case 5: r.b3 &= ~(piece >> (66 - x - (y - 3) * 22)); // fallthrough
      case 3: r.b2 &= ~(piece << (x + (y - 3) * 22)); break;
      case 7: r.b4 &= ~(piece >> (44 - x)); // fallthrough
      case 6: r.b3 &= ~(piece << (x + (y - 6) * 22)); break;
      default: __builtin_unreachable();
    }
    return r;
  }
  constexpr Board PlaceI0_(int x, int y) const {
    Board r = *this;
    y -= 2;
    switch (y) {
      case 0: r.b1 &= ~(kIPiece0a_ << x); r.b2 &= ~(kIPiece0c_ << x); break;
      case 1: r.b1 &= ~(kIPiece0b_ << (x + 22)); r.b2 &= ~(kIPiece0b_ << x); break;
      case 2: r.b1 &= ~(kIPiece0c_ << (x + 44)); r.b2 &= ~(kIPiece0a_ << x); break;
      case 3: r.b2 &= ~(kIPiece0a_ << x); r.b3 &= ~(kIPiece0c_ << x); break;
      case 4: r.b2 &= ~(kIPiece0b_ << (x + 22)); r.b3 &= ~(kIPiece0b_ << x); break;
      case 5: r.b2 &= ~(kIPiece0c_ << (x + 44)); r.b3 &= ~(kIPiece0a_ << x); break;
      case 6: r.b3 &= ~(kIPiece0a_ << x); r.b4 &= ~(kIPiece0c_ << x); break;
      default: __builtin_unreachable();
    }
    return r;
  }
 public:
  static constexpr uint64_t kBoardMask = 0xfffff3ffffcfffffL;
  static constexpr uint32_t kColumnMask = 0xfffff;

  uint64_t b1, b2, b3, b4;

  constexpr int Count() const {
    return 200 - (__builtin_popcountll(b1) + __builtin_popcountll(b2) +
                  __builtin_popcountll(b3) + __builtin_popcountll(b4));
  }

  constexpr void Normalize() {
    b1 &= kBoardMask;
    b2 &= kBoardMask;
    b3 &= kBoardMask;
    b4 &= kColumnMask;
  }

  constexpr uint32_t Column(int c) const {
    switch (c) {
      case 0: case 1: case 2: return b1 >> (c * 22) & kColumnMask;
      case 3: case 4: case 5: return b2 >> ((c - 3) * 22) & kColumnMask;
      case 6: case 7: case 8: return b3 >> ((c - 6) * 22) & kColumnMask;
      case 9: return b4;
    }
    __builtin_unreachable();
  }

  constexpr std::pair<int, Board> ClearLines() const {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    // use an order in favor of vectorization
    // pext will clear unnecessary bits
    uint32_t cols[] = {
        b1, b2, b3, b4,
        b1 >> 22, b2 >> 22, b3 >> 22, 0,
        b1 >> 44, b2 >> 44, b3 >> 44};
#pragma GCC diagnostic pop
    uint32_t linemask = ~(cols[0] & cols[1] & cols[2] & cols[3] & cols[4] &
                          cols[5] & cols[6] & cols[8] & cols[9] & cols[10]) & kColumnMask;
    if (linemask == kColumnMask) return {0, *this};
    int lines = 20 - __builtin_popcount(linemask);
    for (int i = 0; i < 10; i++) {
      cols[i] = _pext_u32(cols[i], linemask) << lines | ((1 << lines) - 1);
    }
    return {lines, {
        cols[0] | (uint64_t)cols[4] << 22 | (uint64_t)cols[8] << 44,
        cols[1] | (uint64_t)cols[5] << 22 | (uint64_t)cols[9] << 44,
        cols[2] | (uint64_t)cols[6] << 22 | (uint64_t)cols[10] << 44,
        cols[3]}};
  }

  // x = 1 or 2
  constexpr Board ShiftLeft(int x) const {
    return {b1 >> (x * 22) | b2 << (66 - x * 22),
            b2 >> (x * 22) | b3 << (66 - x * 22),
            b3 >> (x * 22) | b4 << (66 - x * 22),
            0};
  }
  // x = 1 or 2
  constexpr Board ShiftRight(int x) const {
    return {b1 << (x * 22),
            b2 << (x * 22) | b1 >> (66 - x * 22),
            b3 << (x * 22) | b2 >> (66 - x * 22),
            b3 >> (66 - x * 22) & kColumnMask};
  }
  constexpr Board ShiftUpNoFilter(int x) const {
    return {b1 >> x, b2 >> x, b3 >> x, b4 >> x};
  }
  constexpr Board ShiftDownNoFilter(int x) const {
    return {b1 << x, b2 << x, b3 << x, b4 << x};
  }

  constexpr std::array<Board, 4> TMap() const {
    Board u = ShiftUpNoFilter(1);
    Board d = ShiftDownNoFilter(1);
    Board l = ShiftLeft(1);
    Board r = ShiftRight(1);
    return {{
      u & l & r & *this,
      u & d & r & *this,
      d & l & r & *this,
      d & l & u & *this,
    }};
  }
  constexpr std::array<Board, 4> JMap() const {
    Board u = ShiftUpNoFilter(1);
    Board d = ShiftDownNoFilter(1);
    Board l = ShiftLeft(1);
    Board r = ShiftRight(1);
    Board ul = u.ShiftLeft(1);
    Board ur = u.ShiftRight(1);
    Board dl = d.ShiftLeft(1);
    Board dr = d.ShiftRight(1);
    return {{
      ul & l & r & *this,
      ur & u & d & *this,
      dr & l & r & *this,
      dl & u & d & *this,
    }};
  }
  constexpr std::array<Board, 2> ZMap() const {
    Board u = ShiftUpNoFilter(1);
    Board l = ShiftLeft(1);
    Board r = ShiftRight(1);
    Board ul = u.ShiftLeft(1);
    Board dl = l.ShiftDownNoFilter(1);
    return {{
      u & r & ul & *this,
      u & l & dl & *this,
    }};
  }
  constexpr std::array<Board, 1> OMap() const {
    Board u = ShiftUpNoFilter(1);
    Board r = ShiftRight(1);
    Board ur = u.ShiftRight(1);
    return {{u & r & ur & *this}};
  }
  constexpr std::array<Board, 2> SMap() const {
    Board u = ShiftUpNoFilter(1);
    Board d = ShiftDownNoFilter(1);
    Board l = ShiftLeft(1);
    //Board r = ShiftRight(1);
    Board ur = u.ShiftRight(1);
    Board ul = u.ShiftLeft(1);
    return {{
      u & l & ur & *this,
      d & l & ul & *this,
    }};
  }
  constexpr std::array<Board, 4> LMap() const {
    Board u = ShiftUpNoFilter(1);
    Board d = ShiftDownNoFilter(1);
    Board l = ShiftLeft(1);
    Board r = ShiftRight(1);
    Board ul = u.ShiftLeft(1);
    Board ur = u.ShiftRight(1);
    Board dl = d.ShiftLeft(1);
    Board dr = d.ShiftRight(1);
    return {{
      ur & l & r & *this,
      dr & u & d & *this,
      dl & l & r & *this,
      ul & u & d & *this,
    }};
  }
  constexpr std::array<Board, 2> IMap() const {
    Board u = ShiftUpNoFilter(1);
    Board d = ShiftDownNoFilter(1);
    Board d2 = ShiftDownNoFilter(2);
    Board l = ShiftLeft(1);
    Board r = ShiftRight(1);
    Board r2 = ShiftRight(2);
    return {{
      l & r & r2 & *this,
      u & d & d2 & *this,
    }};
  }

  constexpr Board PlaceT(int r, int x, int y) const {
    switch (r) {
      case 0: return Place3Wide_(kTPiece0_, x, y, 0, 1);
      case 1: return Place2Wide_(kTPiece1_, x, y, 1, 1);
      case 2: return Place3Wide_(kTPiece2_, x, y, 1, 1);
      case 3: return Place2Wide_(kTPiece3_, x, y, 1, 0);
    }
    __builtin_unreachable();
  }
  constexpr Board PlaceJ(int r, int x, int y) const {
    switch (r) {
      case 0: return Place3Wide_(kJPiece0_, x, y, 0, 1);
      case 1: return Place2Wide_(kJPiece1_, x, y, 1, 1);
      case 2: return Place3Wide_(kJPiece2_, x, y, 1, 1);
      case 3: return Place2Wide_(kJPiece3_, x, y, 1, 0);
    }
    __builtin_unreachable();
  }
  constexpr Board PlaceZ(int r, int x, int y) const {
    switch (r) {
      case 0: return Place3Wide_(kZPiece0_, x, y, 0, 1);
      case 1: return Place2Wide_(kZPiece1_, x, y, 1, 0);
    }
    __builtin_unreachable();
  }
  constexpr Board PlaceO(int r, int x, int y) const {
    return Place2Wide_(kOPiece_, x, y, 0, 1);
  }
  constexpr Board PlaceS(int r, int x, int y) const {
    switch (r) {
      case 0: return Place3Wide_(kSPiece0_, x, y, 0, 1);
      case 1: return Place2Wide_(kSPiece1_, x, y, 1, 0);
    }
    __builtin_unreachable();
  }
  constexpr Board PlaceL(int r, int x, int y) const {
    switch (r) {
      case 0: return Place3Wide_(kLPiece0_, x, y, 0, 1);
      case 1: return Place2Wide_(kLPiece1_, x, y, 1, 1);
      case 2: return Place3Wide_(kLPiece2_, x, y, 1, 1);
      case 3: return Place2Wide_(kLPiece3_, x, y, 1, 0);
    }
    __builtin_unreachable();
  }
  constexpr Board PlaceI(int r, int x, int y) const {
    switch (r) {
      case 0: return PlaceI0_(x, y);
      case 1: return Place1Wide_(kIPiece1_, x, y, 2);
    }
    __builtin_unreachable();
  }

  constexpr Board Place(int piece, int r, int x, int y) const {
    switch (piece) {
      case 0: return PlaceT(r, x, y);
      case 1: return PlaceJ(r, x, y);
      case 2: return PlaceZ(r, x, y);
      case 3: return PlaceO(r, x, y);
      case 4: return PlaceS(r, x, y);
      case 5: return PlaceL(r, x, y);
      case 6: return PlaceI(r, x, y);
    }
    __builtin_unreachable();
  }

  constexpr bool operator==(const Board& x) const {
    return b1 == x.b1 && b2 == x.b2 && b3 == x.b3 && b4 == x.b4;
  }
};

constexpr Board operator|(const Board& x, const Board& y) {
  return {x.b1 | y.b1, x.b2 | y.b2, x.b3 | y.b3, x.b4 | y.b4};
}
constexpr Board operator&(const Board& x, const Board& y) {
  return {x.b1 & y.b1, x.b2 & y.b2, x.b3 & y.b3, x.b4 & y.b4};
}
constexpr Board operator~(const Board& x) {
  Board r = {~x.b1, ~x.b2, ~x.b3, ~x.b4};
  r.Normalize();
  return r;
}

uint64_t Hash(uint64_t a, uint64_t b) {
  static const uint64_t table[3] = {0x9e3779b185ebca87, 0xc2b2ae3d27d4eb4f, 0x165667b19e3779f9};
  auto Mix = [](uint64_t a, uint64_t b) {
    a += b * table[1];
    a = (a << 31) | (a >> 33);
    return a * table[0];
  };
  uint64_t v1 = Mix(-table[0], a);
  uint64_t v2 = Mix(table[1], b);
  uint64_t ret = ((v1 << 18) | (v1 >> 46)) + ((v2 << 7) | (v2 >> 57));
  ret ^= ret >> 33;
  ret *= table[1];
  ret ^= ret >> 29;
  ret *= table[2];
  ret ^= ret >> 32;
  return ret;
}

struct BoardHash {
  size_t operator()(const Board& b) const {
    return Hash(Hash(b.b1, b.b3), Hash(b.b2, b.b4));
  }
};

struct Position {
  int r, x, y;
  Position L() const { return {r, x, y - 1}; }
  Position R() const { return {r, x, y + 1}; }
  Position D() const { return {r, x + 1, y}; }
  template <int R> Position A() const { return {r == R - 1 ? 0 : r + 1, x, y}; }
  template <int R> Position B() const { return {r == 0 ? R - 1 : r - 1, x, y}; }
  bool operator<(const Position& p) const { return std::tie(r, x, y) < std::tie(p.r, p.x, p.y); }
  bool operator==(const Position& p) const { return std::tie(r, x, y) == std::tie(p.r, p.x, p.y); }
};
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
          for (int ny : {y - 1, y + 1}) {
            if (Check({r, now_x, ny})) {
              Insert(r, ny, nxt_state);
              if (R > 1 && val.second == 0) {
                if (Check({ra, now_x, ny})) Insert(ra, ny, both_state);
                if (Check({rb, now_x, ny})) Insert(rb, ny, both_state);
              }
            }
          }
        }
        if (R > 1 && val.second == 0) {
          StateTuple nxt_state = Decrement({val.first, frames_per_tap});
          if (Check({ra, now_x, y})) Insert(ra, y, nxt_state);
          if (Check({rb, now_x, y})) Insert(rb, y, nxt_state);
        }
      }
    }
    if (!flag) break;
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
            bs &= ~cur_bs;
            if (bs == 0) continue;
            cur_bs |= bs;
            flag = true;

            int ds1 = Decrement(s1), ds2 = Decrement(s2), df = frames_per_tap - 1;
            InsertCurrent(r, y, ds1, ds2, bs);
            int ra = r == R - 1 ? 0 : r + 1;
            int rb = r == 0 ? R - 1 : r - 1;
            if (s1 == 0) {
              for (int ny : {y - 1, y + 1}) {
                if (Check({r, now_x, ny})) {
                  InsertCurrent(r, ny, df, ds2, bs);
                  if (R > 1 && s2 == 0) {
                    InsertCurrent(ra, ny, df, df, bs);
                    InsertCurrent(rb, ny, df, df, bs);
                  }
                }
              }
            }
            if (R > 1 && s2 == 0) {
              InsertCurrent(ra, y, ds1, df, bs);
              InsertCurrent(rb, y, ds1, df, bs);
            }
          }
        }
      }
    }
    if (!flag) break;
    for (int r = 0; r < R; r++) {
      for (int y = 0; y < 10; y++) {
        if (!Check({r, now_x, y})) {
          memset(next[r][y], 0, sizeof(next[r][y]));
        } else if (to_drop == frames_per_drop && !Check({r, now_x + 1, y})) {
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

#include <cstdio>
#include <chrono>
#include <unordered_map>

struct Edge {
  Position pos;
  std::vector<uint8_t> nxt;
};
struct NodeEdge {
  std::vector<std::pair<int, int>> nexts;
  std::vector<Edge> edges;
};
using EdgeList = std::array<NodeEdge, 7>;
using BoardMap = std::unordered_map<Board, size_t, BoardHash>;

void Print(const Board& b, bool invert = true) {
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 10; j++) printf("%c ", ".1"[(b.Column(j) >> i & 1) ^ invert]);
    puts("");
  }
}

template <int R>
NodeEdge GetEdgeList(const Board& b, int piece, const PositionList<R>& pos_list, const BoardMap& boards) {
  std::bitset<R * 200> tot_bs;
  for (auto& x : pos_list) tot_bs |= x.second;
  if (tot_bs.count() >= 256) throw;
  uint8_t mp[R * 200] = {};
  memset(mp, 0xff, sizeof(mp));
  NodeEdge ret;
  ret.nexts.reserve(tot_bs.count());
  ret.edges.reserve(pos_list.size());
  for (int i = tot_bs._Find_first(); i < tot_bs.size(); i = tot_bs._Find_next(i)) {
    int s = b.Count();
    auto result = b.Place(piece, i / 200, i % 200 / 10, i % 10).ClearLines();
    //result.second.Normalize();
    int t = result.second.Count();
    if ((s+4)%10 != t%10) {
      printf("%d %d (%d,%d,%d)\n", s, t, i/200, i%200/10, i%10);
      Print(b);
      Print(result.second);
    }
    auto it = boards.find(result.second);
    if (it != boards.end()) {
      mp[i] = ret.nexts.size();
      ret.nexts.push_back({it->second, result.first});
    }
  }
  for (auto &[pos, bs] : pos_list) {
    Edge ed = {pos, {}};
    ed.nxt.reserve(bs.count());
    for (int i = bs._Find_first(); i < bs.size(); i = bs._Find_next(i)) {
      if (mp[i] != 0xff) ed.nxt.push_back(mp[i]);
    }
    if (ed.nxt.size()) {
      ed.nxt.shrink_to_fit();
      ret.edges.push_back(std::move(ed));
    }
  }
  ret.nexts.shrink_to_fit();
  ret.edges.shrink_to_fit();
  return std::move(ret);
}

int main() {
  BoardMap boards_mp[5];
  std::vector<Board> boards[5];
  std::vector<EdgeList> edges[5];
  {
    uint8_t buf[25];
    while (fread(buf, 1, 25, stdin) == 25) {
      uint64_t cols[10] = {};
      for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 20; j++) {
          int x = j * 10 + i;
          cols[i] |= (uint64_t)(buf[x / 8] >> (x % 8) & 1) << j;
        }
      }
      Board r{cols[0] | cols[1] << 22 | cols[2] << 44,
              cols[3] | cols[4] << 22 | cols[5] << 44,
              cols[6] | cols[7] << 22 | cols[8] << 44,
              cols[9]};
      int cnt = r.Count();
      if (cnt & 1) continue;
      int group = cnt % 10 / 2;
      boards_mp[group][r] = boards[group].size();
      boards[group].push_back(r);
    }
  }
  int edc = 0, nxtc = 0, adjc = 0, c = 0, cc = 0, p = 0, pp = 0;
  auto start = std::chrono::steady_clock::now();
  auto prev = start;
  {
    for (int group = 0; group < 5; group++) {
      int nxt_group = (group + 2) % 5;
      auto& nxt_map = boards_mp[nxt_group];
      // T, J, Z, O, S, L, I
      for (auto& board : boards[group]) {
        edges[group].emplace_back();
        auto& eds = edges[group].back();
        eds[0] = GetEdgeList<4>(board, 0, SearchMoves<4, 2, 5, 21>(board.TMap()), nxt_map);
        eds[1] = GetEdgeList<4>(board, 1, SearchMoves<4, 2, 5, 21>(board.JMap()), nxt_map);
        eds[2] = GetEdgeList<2>(board, 2, SearchMoves<2, 2, 5, 21>(board.ZMap()), nxt_map);
        eds[3] = GetEdgeList<1>(board, 3, SearchMoves<1, 2, 5, 21>(board.OMap()), nxt_map);
        eds[4] = GetEdgeList<2>(board, 4, SearchMoves<2, 2, 5, 21>(board.SMap()), nxt_map);
        eds[5] = GetEdgeList<4>(board, 5, SearchMoves<4, 2, 5, 21>(board.LMap()), nxt_map);
        eds[6] = GetEdgeList<2>(board, 6, SearchMoves<2, 2, 5, 21>(board.IMap()), nxt_map);
        bool flag = false;
        for (auto& i : eds) {
          edc += i.edges.size();
          nxtc += i.nexts.size();
          for (auto& j : i.edges) adjc += j.nxt.size();
          c++;
          if (i.nexts.size()) cc++, flag = true;
        }
        p++;
        if (flag) pp++;
        if (p % 16384 == 0) {
          auto end = std::chrono::steady_clock::now();
          std::chrono::duration<double> dur = end - start;
          std::chrono::duration<double> dur2 = end - prev;
          printf("%d %d %d %d %d %d %d, %lf / %lf item/s\n", p, pp, c, cc, edc, nxtc, adjc, p / dur.count(), 16384 / dur2.count());
          prev = end;
        }
      }
    }
  }
  printf("%d %d %d %d %d %d %d\n", p, pp, c, cc, edc, nxtc, adjc);
}
