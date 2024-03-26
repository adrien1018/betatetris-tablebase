#pragma once

#include "move_search_no_tmpl.h"

namespace move_search {

template <
    int tap1, int tap2, int tap3, int tap4, int tap5,
    int tap6, int tap7, int tap8, int tap9, int tap10>
class TapTable {
  int t[10];
 public:
  constexpr TapTable() : t{tap1, tap2, tap3, tap4, tap5, tap6, tap7, tap8, tap9, tap10} {
    static_assert(
        tap1 >= 0 && tap2 >= 2 && tap3 >= 2 && tap4 >= 2 && tap5 >= 2 &&
        tap6 >= 2 && tap7 >= 2 && tap8 >= 2 && tap9 >= 2 && tap10 >= 2,
        "each tap must be at least 2 frames apart");
    for (int i = 1; i < 10; i++) t[i] += t[i - 1];
  }
  constexpr int operator[](int x) const { return t[x]; }
  constexpr int* data() { return t; }
  constexpr const int* data() const { return t; }
};

template <int R>
constexpr bool Contains(const std::array<Board, R>& board, const std::array<Board, R>& mask) {
  bool ret = true;
  for (int i = 0; i < R; i++) ret &= (board[i] & mask[i]) == mask[i];
  return ret;
}

template <class T> struct IsTapTable : std::false_type {};
template <int... args> struct IsTapTable<TapTable<args...>> : std::true_type {};

template <int R>
struct TableEntry {
  // Each entry corresponds to a (rot, col) that is possible to reach
  // `num_taps` is the minimum number of taps needed to reach that (rot, col)
  //   (so taps[num_taps-1] will be the exact frame when the piece is moved to (rot, col))
  // This (rot, col) is only reachable if the cells corresponding to `masks` in entry `prev`
  //   (prev is regarded recursively; that is, the `prev` of `prev`, etc., should also be included)
  //   and `masks_nodrop` in this entry are all empty
  //   (`masks_nodrop` should be a subset of `masks` in the same entry)
  uint8_t rot, col, prev, num_taps;
  // `cannot_finish` means no further input is possible since it will be locked at the bottom of the board
  // when it happens, `masks` should not be used
  bool cannot_finish;
  std::array<Board, R> masks, masks_nodrop;

  bool operator<=>(const TableEntry<R>&) const = default;
};

template <int R, class Taps>
constexpr int Phase1TableGen(
    Level level, int initial_frame, int initial_rot, int initial_col,
    TableEntry<R> entries[]) {
#ifndef _MSC_VER
  static_assert(IsTapTable<Taps>::value);
#endif
  constexpr Taps taps{};
  return Phase1TableGen<R>(level, taps.data(), initial_frame, initial_rot, initial_col, entries);
}

template <Level level, int R, int adj_frame, class Taps>
struct Phase1Table {
  static constexpr int initial_N = Phase1TableGen<R, Taps>(
      level, 0, 0, Position::Start.y, std::array<TableEntry<R>, 10*R>().data());
  TableEntry<R> initial[initial_N];
  int adj_N[initial_N];
  TableEntry<R> adj[initial_N][10 * R];
  constexpr Phase1Table() : initial{}, adj_N{}, adj{} {
    constexpr Taps taps{};
    Phase1TableGen<R, Taps>(level, 0, 0, Position::Start.y, initial);
    for (int i = 0; i < initial_N; i++) {
      int frame_start = std::max(adj_frame, taps[initial[i].num_taps]);
      adj_N[i] = Phase1TableGen<R, Taps>(
          level, frame_start, initial[i].rot, initial[i].col, adj[i]);
    }
  }
};

template <Level level, int R, int adj_frame, class Taps>
class Search {
  static constexpr Taps taps{};
  static constexpr Phase1Table<level, R, adj_frame, Taps> table{};

  // N = is_adj ? table.adj_N[X] : initial_N
  // table = is_adj ? table.adj[X] : table.initial
  // initial_frame = is_adj ? <the last> : 0
  template <bool is_adj, int initial_id = 0> constexpr int DoOneSearch(
      const std::array<Board, R>& board, const Column cols[R][10],
      const TuckMasks<R> tuck_masks,
      bool can_adj[],
      Position* positions) const {
    constexpr int total_frames = GetLastFrameOnRow(19, level) + 1;
    constexpr int N = is_adj ? table.adj_N[initial_id] : table.initial_N;
    constexpr int initial_frame = is_adj ? std::max(adj_frame, taps[table.initial[initial_id].num_taps]) : 0;
    if (initial_frame >= total_frames) return 0;

    int sz = 0;
    // phase 1
    bool can_continue[R * 10] = {}; // whether the next tap can continue
    Frames can_tuck_frame_masks[R][10] = {}; // frames that can start a tuck
    Column lock_positions_without_tuck[R][10] = {};

    bool phase_2_possible = false;
    bool can_reach[std::max(N, 1)] = {};
    For<N>([&](auto i_obj) {
      constexpr int i = i_obj.value;
      constexpr auto& entry = is_adj ? table.adj[initial_id][i] : table.initial[i];
      if (i && !can_continue[entry.prev]) return;
      if (!Contains<R>(board, entry.masks_nodrop)) return;
      if (!entry.cannot_finish && Contains<R>(board, entry.masks)) {
        can_continue[i] = true;
      }
      can_reach[i] = true;
    });
#pragma GCC unroll 0
    for (int i = 0; i < N; i++) {
      if (!can_reach[i]) continue;
      auto& entry = is_adj ? table.adj[initial_id][i] : table.initial[i];
      CheckOneInitial<R>(
          level, adj_frame, taps, is_adj, total_frames, initial_frame, entry, cols,
          lock_positions_without_tuck, can_tuck_frame_masks,
          sz, positions, can_adj[i], phase_2_possible);
    }
    if (phase_2_possible) {
      SearchTucks<R>(level, cols, tuck_masks, lock_positions_without_tuck, can_tuck_frame_masks, sz, positions);
    }
    return sz;
  }

 public:
  /** drop sequence:
  *
  * initial phase 1         adj phase 1
  * vvvvvvvvvvv               vvvvvvv
  * L - L - L - - - - - - - - R - R - - - - - - -<lock>
  *               \           ^ adj_frame   \
  *                \A R - - -<lock>          \B R - - -<lock>
  *                 ^^^^                      ^^^^
  *                initial phase 2 (tuck)    adj phase 2
  */
  PossibleMoves MoveSearch(const std::array<Board, R>& board) const {
    constexpr int initial_N = decltype(table)::initial_N;
    Column cols[R][10] = {};
    auto tuck_masks = GetTuckMasks<R>(GetColsAndFrameMasks<R>(level, board, cols));
    bool can_adj[initial_N] = {}; // whether adjustment starting from this (rot, col) is possible

    PossibleMoves ret;
    Position buf[256];
    ret.non_adj.assign(buf, buf + DoOneSearch<false>(
        board, cols, tuck_masks, can_adj, buf));

    For<initial_N>([&](auto i_obj) {
      constexpr int initial_id = i_obj.value;
      auto& entry = table.initial[initial_id];
      if (!can_adj[initial_id]) return;
      if (int x = DoOneSearch<true, initial_id>(board, cols, tuck_masks, can_adj, buf); x) {
        int row = GetRow(std::max(adj_frame, taps[entry.num_taps]), level);
        ret.adj.emplace_back(Position{entry.rot, row, entry.col}, std::vector<Position>(buf, buf + x));
      }
    });
    return ret;
  }
};

} // namespace move_search

template <Level level, int R, int adj_frame, class Taps>
NOINLINE PossibleMoves MoveSearch(const std::array<Board, R>& board) {
  constexpr move_search::Search<level, R, adj_frame, Taps> search{};
  return search.MoveSearch(board);
}

template <Level level, int adj_frame, class Taps>
PossibleMoves MoveSearch(const Board& b, int piece) {
#define ONE_CASE(x) \
    case x: return MoveSearch<level, Board::NumRotations(x), adj_frame, Taps>(b.PieceMap<x>());
  DO_PIECE_CASE(piece);
#undef ONE_CASE
}

template <int adj_frame, class Taps>
PossibleMoves MoveSearch(const Board& b, Level level, int piece) {
#define LEVEL_CASE_TMPL_ARGS ,adj_frame,Taps
  DO_LEVEL_CASE(MoveSearch, b, piece);
#undef LEVEL_CASE_TMPL_ARGS
}

using Tap30Hz = move_search::TapTable<0, 2, 2, 2, 2, 2, 2, 2, 2, 2>;
using Tap20Hz = move_search::TapTable<0, 3, 3, 3, 3, 3, 3, 3, 3, 3>;
using Tap15Hz = move_search::TapTable<0, 4, 4, 4, 4, 4, 4, 4, 4, 4>;
using Tap12Hz = move_search::TapTable<0, 5, 5, 5, 5, 5, 5, 5, 5, 5>;
