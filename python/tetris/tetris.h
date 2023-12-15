#pragma once

#include <random>
#include "../../src/tetris.h"
#include "python.h"

class PythonTetris {
 public:
  PyObject_HEAD
  static constexpr double kInvalidReward_ = -0.3;
  static constexpr double kRewardMultiplier_ = 1e-5; // 10 per maxout
  static constexpr double kBottomMultiplier_ = 1.2;
  double step_reward_ = 5e-4;

 private:
  static constexpr int kTransitionProb_[kPieces][kPieces] = {
  // T  J  Z  O  S  L  I (next)
    {1, 5, 6, 5, 5, 5, 5}, // T (current)
    {6, 1, 5, 5, 5, 5, 5}, // J
    {5, 6, 1, 5, 5, 5, 5}, // Z
    {5, 5, 5, 2, 5, 5, 5}, // O
    {5, 5, 5, 5, 2, 5, 5}, // S
    {6, 5, 5, 5, 5, 1, 5}, // L
    {5, 5, 5, 5, 6, 5, 1}, // I
  };

  std::mt19937_64 rng_;
  int next_piece_;

  int GenNextPiece_(int piece) {
    return std::discrete_distribution<int>(
        kTransitionProb_[piece], kTransitionProb_[piece] + kPieces)(rng_);
  }

 public:
  Tetris tetris;

  PythonTetris(size_t seed) : rng_(seed) {
    Reset(Board::Ones, 0);
  }

  void Reset(const Board& b) {
    int lines = b.Count() % 4 != 0;
    lines += std::uniform_int_distribution<int>(0, kLineCap / 2 - 1)(rng_) * 2;
    Reset(b, lines);
  }

  void Reset(const Board& b, int lines) {
    int first_piece = std::uniform_int_distribution<int>(0, kPieces - 1)(rng_);
    next_piece_ = GenNextPiece_(first_piece);
    tetris.Reset(b, lines, first_piece, next_piece_);
    next_piece_ = GenNextPiece_(next_piece_);
  }

  void Reset(const Board& b, int lines, int now_piece, int next_piece) {
    tetris.Reset(b, lines, now_piece, next_piece);
    next_piece_ = GenNextPiece_(next_piece);
  }

  std::pair<double, double> InputPlacement(const Position& pos) {
    auto [score, lines] = tetris.InputPlacement(pos, next_piece_);
    if (score == -1) return {kInvalidReward_, 0.0f};
    double reward = score * kRewardMultiplier_;
    double n_reward = reward;
    if (lines == 4 && pos.x >= 18) n_reward *= kBottomMultiplier_;
    if (!tetris.IsAdj()) {
      next_piece_ = GenNextPiece_(next_piece_);
      n_reward += step_reward_;
    }
    return {n_reward, reward};
  }

  struct State {
    std::array<std::array<std::array<float, 10>, 20>, 6> board;
    std::array<float, 28> meta;
    std::array<std::array<std::array<float, 10>, 20>, 14> moves;
    std::array<float, 28> move_meta;
    std::array<int, 2> meta_int;
  };

  void GetState(State& state) {
    // board: shape (2, 20, 10) [board, one]
    // meta: shape (28,) [group(5), now_piece(7), next_piece(7), is_adj(1), hz(4), adj(4)]
    // meta_int: shape (2,) [entry, now_piece]
    // moves: shape (10, 20, 10) [board, one, moves(4), adj_moves(4)]
    // move_meta: shape (28,) [speed(4), to_transition(21), (level-18)*0.1, lines*0.01, pieces*0.004]
    {
      auto byte_board = tetris.GetBoard().ToByteBoard();
      for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) state.board[0][i][j] = byte_board[i][j];
        for (int j = 0; j < 10; j++) state.board[1][i][j] = 1;
        for (int j = 0; j < 10; j++) state.moves[0][i][j] = byte_board[i][j];
        for (int j = 0; j < 10; j++) state.moves[1][i][j] = 1;
      }
      auto& move_map = tetris.GetPossibleMoveMap();
      for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 20; i++) {
          for (int j = 0; j < 10; j++) state.moves[2 + r][i][j] = move_map[r][i][j] ? 1 : 0;
          for (int j = 0; j < 10; j++) state.moves[6 + r][i][j] = move_map[r][i][j] == 2;
        }
        memset(state.board.data() + (2 + r), 0, sizeof(state.board[0]));
        memset(state.moves.data() + (10 + r), 0, sizeof(state.moves[0]));
      }
    }
    if (tetris.IsAdj()) {
      auto pos = tetris.InitialMove();
      state.board[2 + pos.r][pos.x][pos.y] = 1;
      state.moves[10 + pos.r][pos.x][pos.y] = 1;
    }

    memset(state.meta.data(), 0, sizeof(state.meta));
    state.meta[0 + tetris.GetBoard().Count() / 2 % 5] = 1;
    state.meta[5 + tetris.NowPiece()] = 1;
    if (tetris.IsAdj()) {
      state.meta[12 + tetris.NextPiece()] = 1;
      state.meta[19] = 1;
    }
    state.meta[20] = 1; // hardcode now; modify if extended
    state.meta[24] = 1;

    int lines = tetris.GetLines();
    state.meta_int[0] = lines / 2;
    state.meta_int[1] = tetris.NowPiece();

    memset(state.move_meta.data(), 0, sizeof(state.move_meta));
    int to_transition = 0;
    switch (tetris.LevelSpeed()) {
      case kLevel18: state.move_meta[0] = 1; to_transition = std::min(130, kLineCap) - lines; break;
      case kLevel19: state.move_meta[1] = 1; to_transition = std::min(230, kLineCap) - lines; break;
      case kLevel29: state.move_meta[2] = 1; to_transition = std::min(330, kLineCap) - lines; break;
      case kLevel39: state.move_meta[3] = 1; to_transition = kLineCap - lines; break;
    }
    if (to_transition <= 10) { // 4..13
      state.move_meta[4 + (to_transition - 1)] = 1;
    } else if (to_transition <= 22) { // 14..17
      state.move_meta[14 + (to_transition - 11) / 3] = 1;
    } else if (to_transition <= 40) { // 18..20
      state.move_meta[18 + (to_transition - 22) / 6] = 1;
    } else if (to_transition <= 60) { // 21,22
      state.move_meta[21 + (to_transition - 40) / 10] = 1;
    } else {
      state.move_meta[23] = 1;
    }
    state.move_meta[24] = to_transition * 0.01;
    state.move_meta[25] = (tetris.GetLevel() - 18) * 0.1;
    state.move_meta[26] = tetris.GetLines() * 0.01;
    state.move_meta[27] = tetris.GetPieces() * 0.004;
  }

  operator Tetris() const { return tetris; }
};

extern PyTypeObject py_tetris_class;
