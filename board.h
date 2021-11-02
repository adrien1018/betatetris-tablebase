#ifndef BOARD_H_
#define BOARD_H_

#include <cstdint>
#include <cstring>
#include <array>

#include <immintrin.h>

class alignas(32) Board;
constexpr Board operator|(const Board& x, const Board& y);
constexpr Board operator&(const Board& x, const Board& y);

constexpr int kBoardBytes = 25;

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

  static Board FromBytes(const uint8_t buf[kBoardBytes]) {
    constexpr uint64_t kMask3 = 0x701C0701C0701C07L;
    constexpr uint64_t kMask2 = 0x300C0300C0300C03L;
    constexpr uint64_t kColMask3 = 0x249249249249249L;
    constexpr uint64_t kColMask2 = 0x5555555555L;
    uint64_t cur = *(uint64_t*)buf;
    uint64_t r1 = _pext_u64(cur, kMask3); // 7,7,7
    uint64_t r2 = _pext_u64(cur, kMask3 << 3); // 7,6,6
    uint64_t r3 = _pext_u64(cur, kMask2 << 6); // 6
    uint64_t r4 = _pext_u64(cur, kMask2 << 8); // 6
    cur = *(uint64_t*)(buf + 8);
    r1 |= _pext_u64(cur, kMask3 << 6) << 21; // 6,6,6
    r2 |= _pext_u64(cur, kMask3 >> 1) << 19; // 6,7,7
    r3 |= _pext_u64(cur, kMask2 << 2) << 12; // 7
    r4 |= _pext_u64(cur, kMask2 << 4) << 12; // 6
    cur = *(uint64_t*)(buf + 16);
    r1 |= _pext_u64(cur, kMask3 << 2) << 39; // 7,7,6
    r2 |= _pext_u64(cur, kMask3 << 5) << 39; // 6,6,6
    r3 |= _pext_u64(cur, kMask2 << 8) << 26; // 6
    r4 |= _pext_u64(cur, kMask2) << 24; // 7
    r1 |= (uint64_t)(buf[24] & 0x1) << 59;
    r2 |= (uint64_t)(buf[24] & 0xe) << (57 - 1);
    r3 |= (uint64_t)(buf[24] & 0x30) << (38 - 4);
    r4 |= (uint64_t)(buf[24] & 0xc0) << (38 - 6);
    return {_pext_u64(r1, kColMask3) | _pext_u64(r1, kColMask3 << 1) << 22 | _pext_u64(r1, kColMask3 << 2) << 44,
            _pext_u64(r2, kColMask3) | _pext_u64(r2, kColMask3 << 1) << 22 | _pext_u64(r2, kColMask3 << 2) << 44,
            _pext_u64(r3, kColMask2) | _pext_u64(r3, kColMask2 << 1) << 22 | _pext_u64(r4, kColMask2) << 44,
            _pext_u64(r4, kColMask2 << 1)};
  }

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

  void ToBytes(uint8_t buf[kBoardBytes]) const {
    constexpr uint64_t kMask3 = 0x701C0701C0701C07L;
    constexpr uint64_t kMask2 = 0x300C0300C0300C03L;
    constexpr uint64_t kColMask3 = 0x249249249249249L;
    constexpr uint64_t kColMask2 = 0x5555555555L;
    uint64_t r1 = _pdep_u64(b1, kColMask3) | _pdep_u64(b1 >> 22, kColMask3 << 1) | _pdep_u64(b1 >> 44, kColMask3 << 2);
    uint64_t r2 = _pdep_u64(b2, kColMask3) | _pdep_u64(b2 >> 22, kColMask3 << 1) | _pdep_u64(b2 >> 44, kColMask3 << 2);
    uint64_t r3 = _pdep_u64(b3, kColMask2) | _pdep_u64(b3 >> 22, kColMask2 << 1);
    uint64_t r4 = _pdep_u64(b3 >> 44, kColMask2) | _pdep_u64(b4, kColMask2 << 1);
    buf[24] = (r1 >> 59 & 0x1) | (r2 >> (57 - 1) & 0xe) |
              (r3 >> (38 - 4) & 0x30) | (r4 >> (38 - 6) & 0xc0);
    uint64_t* cur = (uint64_t*)(buf + 16);
    *cur  = _pdep_u64(r1 >> 39, kMask3 << 2);
    *cur |= _pdep_u64(r2 >> 39, kMask3 << 5);
    *cur |= _pdep_u64(r3 >> 26, kMask2 << 8);
    *cur |= _pdep_u64(r4 >> 24, kMask2);
    cur = (uint64_t*)(buf + 8);
    *cur  = _pdep_u64(r1 >> 21, kMask3 << 6);
    *cur |= _pdep_u64(r2 >> 19, kMask3 >> 1);
    *cur |= _pdep_u64(r3 >> 12, kMask2 << 2);
    *cur |= _pdep_u64(r4 >> 12, kMask2 << 4);
    cur = (uint64_t*)buf;
    *cur  = _pdep_u64(r1, kMask3);
    *cur |= _pdep_u64(r2, kMask3 << 3);
    *cur |= _pdep_u64(r3, kMask2 << 6);
    *cur |= _pdep_u64(r4, kMask2 << 8);
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
    uint32_t linemask = (cols[0] | cols[1] | cols[2] | cols[3] | cols[4] |
                         cols[5] | cols[6] | cols[8] | cols[9] | cols[10]) & kColumnMask;
    if (linemask == kColumnMask) return {0, *this};
    int lines = 20 - __builtin_popcount(linemask);
    for (int i = 0; i < 11; i++) {
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

#endif // BOARD_H_
