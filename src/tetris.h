#pragma once

#include <cstring>
#include "game.h"
#include "move_search.h"

class Tetris {
 public:
  using MoveMap = std::array<ByteBoard, 4>;
  static constexpr uint8_t kNoAdj = 1;
  static constexpr uint8_t kHasAdj = 2;

 private:
  Board board_;
  int lines_;
  int pieces_;
  bool is_adj_;
  int initial_move_;
  int now_piece_;
  int next_piece_;
  bool game_over_;
  PossibleMoves moves_;
  MoveMap move_map_;

  // stats
  int run_score_;
  int run_lines_;
  int run_pieces_;

  void CalculateMoves_(bool regenerate) {
    if (regenerate) {
      moves_ = MoveSearch<ADJ_DELAY, TAP_SPEED>(board_, LevelSpeed(), now_piece_);
      if (moves_.non_adj.empty() && moves_.adj.empty()) {
        game_over_ = true;
        return;
      }
    }
    memset(move_map_.data(), 0, sizeof(move_map_));
    if (!is_adj_) {
      for (auto& i : moves_.non_adj) move_map_[i.r][i.x][i.y] = kNoAdj;
      for (auto& [i, _] : moves_.adj) move_map_[i.r][i.x][i.y] = kHasAdj;
    } else {
      for (auto& i : moves_.adj[initial_move_].second) move_map_[i.r][i.x][i.y] = kNoAdj;
    }
  }

 public:
  void Reset(const Board& b, int lines, int now_piece, int next_piece) {
    int pieces = (lines * 10 + b.Count()) / 4;
    if (pieces * 4 != lines * 10 + (int)b.Count()) throw std::runtime_error("Incorrect lines");
    board_ = b;
    lines_ = lines;
    pieces_ = pieces;
    is_adj_ = false;
    initial_move_ = 0;
    now_piece_ = now_piece;
    next_piece_ = next_piece;
    game_over_ = false;
    CalculateMoves_(true);
    run_score_ = 0;
    run_lines_ = 0;
    run_pieces_ = 0;
  }

  // (score, lines)
  // score == -1 if invalid
  std::pair<int, int> InputPlacement(const Position& pos, int next_piece) {
    if (game_over_) throw std::logic_error("already game over");
    if (next_piece < 0 || next_piece >= (int)kPieces) throw std::range_error("Invalid piece");
    uint8_t location = move_map_[pos.r][pos.x][pos.y];
    if (!location) return {-1, 0};
    if (location == kNoAdj) {
      auto before_clear = board_.Place(now_piece_, pos.r, pos.x, pos.y);
      // do not allow placing pieces to be cut off from the board
      if (board_.Count() + 4 != before_clear.Count()) return {-1, 0};

      auto [lines, new_board] = before_clear.ClearLines();
      int delta_score = Score(lines, GetLevel());
      board_ = new_board;
      pieces_++;
      lines_ += lines;
      is_adj_ = false;
      initial_move_ = 0;
      now_piece_ = next_piece_;
      next_piece_ = next_piece;
      if (lines_ >= kLineCap) {
        game_over_ = true;
      } else {
        CalculateMoves_(true);
      }
      run_score_ += delta_score;
      run_lines_ += lines;
      run_pieces_++;
      return {delta_score, lines};
    } else {
      for (size_t i = 0; i < moves_.adj.size(); i++) {
        if (moves_.adj[i].first == pos) {
          initial_move_ = i;
          break;
        }
      }
      is_adj_ = true;
      CalculateMoves_(false);
      return {0, 0};
    }
  }

  void SetNextPiece(int piece) {
    if (piece < 0 || piece >= (int)kPieces) throw std::range_error("Invalid piece");
    next_piece_ = piece;
  }

  const MoveMap& GetPossibleMoveMap() const { return move_map_; }
  const Board& GetBoard() const { return board_; }

  int GetLevel() const { return GetLevelByLines(lines_); }
  Level LevelSpeed() const { return GetLevelSpeed(GetLevel()); }
  bool IsAdj() const { return is_adj_; }
  int GetPieces() const { return pieces_; }
  int GetLines() const { return lines_; }
  int NowPiece() const { return now_piece_; }
  int NextPiece() const { return next_piece_; }
  bool IsOver() const { return game_over_; }
  Position InitialMove() const {
    if (!is_adj_) throw std::logic_error("No initial move");
    return moves_.adj[initial_move_].first;
  }

  int RunPieces() const { return run_pieces_; }
  int RunLines() const { return run_lines_; }
  int RunScore() const { return run_score_; }
};
