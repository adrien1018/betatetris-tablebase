#include <array>
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
    bool flag = true;
    for (int y = 0; y < kM; y++) flag &= field[i][y];
    if (flag) {
      j++;
    } else if (i != j) {
      field[j] = field[i];
    }
  }
  int ans = j + 1;
  for (; j >= 0; j--) std::fill(field[j].begin(), field[j].end(), true);
  return ans;
}
