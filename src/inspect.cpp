#include "inspect.h"

#include <ranges>
#include <iostream>
#include <string_view>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop

#include "io.h"
#include "edge.h"
#include "board.h"
#include "files.h"

template<> struct fmt::formatter<Position> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) const { return ctx.begin(); }
  template <typename FormatContext>
  auto format(const Position& pos, FormatContext& ctx) const {
    return format_to(ctx.out(), "({},{},{})", pos.r, pos.x, pos.y);
  }
};

namespace {

void PrintGrid(const std::vector<std::string>& grid, int col_width = 12, int cols = 4) {
  for (size_t i = 0; i < grid.size(); i += cols) {
    size_t N = std::min((size_t)cols, grid.size() - i);
    std::vector<std::vector<std::string_view>> vec;
    size_t rows = 0;
    for (size_t j = i; j < i + N; j++) {
      vec.emplace_back();
      for (const auto k : std::views::split(std::string_view(grid[j]), std::string_view{"\n"})) {
        vec.back().emplace_back(k.begin(), k.end());
      }
      rows = std::max(rows, vec.back().size());
    }
    for (size_t r = 0; r < rows; r++) {
      size_t len = 0;
      for (size_t j = 0; j < N; j++) {
        size_t col_end = (j + 1) * col_width;
        if (vec[j].size() >= rows - r) {
          auto& item = vec[j][r + vec[j].size() - rows];
          len += item.size();
          std::cout << item;
        }
        if (len < col_end && j != N - 1) {
          std::cout << std::string(col_end - len, ' ');
          len = col_end;
        }
      }
      std::cout << '\n';
    }
  }
}

} // namespace

void InspectBoard(int group, const std::vector<long>& board_idx) {
  ClassReader<CompactBoard> reader(BoardPath(group));
  for (auto id : board_idx) {
    reader.Seek(id, 4096);
    try {
      auto board = Board(reader.ReadOne(4096));
      std::cout << fmt::format("Group {}, board {}:\n{{{:#x}, {:#x}, {:#x}, {:#x}}}\n",
                              group, id, board.b1, board.b2, board.b3, board.b4)
                << board.ToString();
    } catch (ReadError&) {
      std::cout << fmt::format("Group {}, board {} not found\n", group, id);
    }
  }
}

void InspectEdge(int group, const std::vector<long>& board_idx, Level level, int piece) {
  int level_int = static_cast<int>(level);
  ClassReader<CompactBoard> reader_cur(BoardPath(group));
  ClassReader<CompactBoard> reader_nxt(BoardPath((group + 2) % 5));
  CompressedClassReader<EvaluateNodeEdges> reader_eval_ed(EvaluateEdgePath(group, level_int));
  CompressedClassReader<PositionNodeEdges> reader_pos_ed(PositionEdgePath(group, level_int));
  for (auto id : board_idx) {
    reader_cur.Seek(id, 4096);
    reader_eval_ed.Seek(id * kPieces + piece, 0, 0);
    reader_pos_ed.Seek(id * kPieces + piece, 0, 0);
    auto board = Board(reader_cur.ReadOne(4096));
    auto eval_ed = reader_eval_ed.ReadOne(0, 0);
    auto pos_ed = reader_pos_ed.ReadOne(0, 0);

    std::cout << fmt::format("Group {}, board {}:\n", group, id) << board.ToString();
    std::vector<std::string> next_boards;
    for (size_t i = 0; i < eval_ed.next_ids.size(); i++) {
      auto [nxt, lines] = eval_ed.next_ids[i];
      reader_nxt.Seek(nxt, 0);
      auto nboard = Board(reader_nxt.ReadOne(0));
      next_boards.push_back(nboard.ToString(false, true, false));
      const auto& pos = pos_ed.nexts[i];
      next_boards.back() += fmt::format("{},{} {},{},{}\n{}", i, lines, pos.r, pos.x, pos.y, nxt);
    }
    std::cout << "Nexts:\n";
    PrintGrid(next_boards);
    std::vector<Position> non_adj_pos;
    for (auto& i : eval_ed.non_adj) non_adj_pos.push_back(pos_ed.nexts[i]);
    std::cout << fmt::format("Non-adj: {} {}\n", eval_ed.non_adj, non_adj_pos);
    std::cout << "Adjs:\n";
    if (eval_ed.use_subset) {
      std::cout << fmt::format("({} before expanding)\n", eval_ed.subset_idx_prev.size());
      eval_ed.CalculateAdj();
    }
    for (size_t i = 0; i < eval_ed.adj.size(); i++) {
      std::cout << fmt::format("{}: {}\n", pos_ed.adj[i], eval_ed.adj[i]);
    }
  }
}
