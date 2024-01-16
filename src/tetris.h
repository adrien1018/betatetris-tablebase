#pragma once

#include <cstring>
#include "game.h"
#include "move_search.h"
#include "frame_sequence.h"

class Tetris {
 public:
  using MoveMap = std::array<ByteBoard, 4>;
  static constexpr uint8_t kNoAdj = 1;
  static constexpr uint8_t kHasAdj = 2;

  enum AgentMode {
    kNormalAgent,
    kSingleAgent,
    kPushdownAgent
  };

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
  int consecutive_fail_;
  AgentMode agent_mode_;
  bool cur_nnb_;

  // stats
  int run_score_;
  int run_lines_;
  int run_pieces_;

  static bool ShouldPushdown_(int lines, int cleared) {
    int start_level = GetLevelByLines(lines), end_level = GetLevelByLines(lines + cleared);
    bool is_transition = start_level != end_level;
    return (is_transition && end_level == 158) || (end_level == 253);
  }

  static bool ShouldNNB_(int lines, int cleared) {
    int start_level = GetLevelByLines(lines), end_level = GetLevelByLines(lines + cleared);
    bool is_transition = start_level != end_level;
    return (is_transition && (end_level == 155 || end_level == 156 || end_level == 159)) ||
           (!is_transition && cleared && (end_level == 157 || end_level == 158)) ||
           (end_level == 249 || end_level == 255);
  }

  int GetClearLines_(const Position& pos) const {
    return board_.Place(now_piece_, pos.r, pos.x, pos.y).ClearLines().first;
  }

  int GetAdjDelay_() const {
    return agent_mode_ == kNormalAgent ? ADJ_DELAY : 61;
  }

  void SetAgentMode_() {
    int level = GetLevelByLines(lines_);
    if ((level >= 153 && level < 158) || (level >= 161 && level < 174) || (level >= 248 && level < 254)) {
      // switch agent at clean and lower stack
      if (agent_mode_ == kNormalAgent && board_.Height() <= 6 && board_.NumOverhang() <= 1) {
        if (level >= 161 && level < 174) {
          agent_mode_ = kSingleAgent;
        } else {
          agent_mode_ = kPushdownAgent;
        }
      }
    } else if (level >= 174 && lines_ < 248) {
      // switch agent at very clean and low stack
      if (agent_mode_ == kSingleAgent && board_.Height() <= 4 && board_.NumOverhang() == 0) agent_mode_ = kNormalAgent;
    } else {
      agent_mode_ = kNormalAgent;
    }
  }

  void CalculateMoves_(bool regenerate) {
    if (regenerate) {
      if (agent_mode_ == kNormalAgent) {
        moves_ = MoveSearch<ADJ_DELAY, TAP_SPEED>(board_, LevelSpeed(), now_piece_);
      } else {
        moves_ = MoveSearch<61, TAP_SPEED>(board_, LevelSpeed(), now_piece_);
      }
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

  std::pair<int, int> StepGame_(const Position& pos, int next_piece) {
    auto before_clear = board_.Place(now_piece_, pos.r, pos.x, pos.y);
    // do not allow placing pieces to be cut off from the board
    if (board_.Count() + 4 != before_clear.Count()) {
      consecutive_fail_++;
      return {-1, 0};
    }

    auto [clear_lines, new_board] = before_clear.ClearLines();
    cur_nnb_ = ShouldNNB_(lines_, clear_lines);
    if (ShouldPushdown_(lines_, clear_lines)) run_score_++;
    lines_ += clear_lines;
    int delta_score = Score(clear_lines, GetRealLevel());
    board_ = new_board;
    pieces_++;
    is_adj_ = false;
    initial_move_ = 0;
    now_piece_ = next_piece_;
    next_piece_ = next_piece;
    SetAgentMode_();
    if (lines_ >= kLineCap) {
      game_over_ = true;
    } else {
      CalculateMoves_(true);
    }
    consecutive_fail_ = 0;
    run_score_ += delta_score;
    run_lines_ += clear_lines;
    run_pieces_++;
    return {delta_score, clear_lines};
  }

  static void AddStartNNB_(FrameSequence& seq) {
    for (size_t i = seq.size(); i > 0; i--) {
      if (!seq[i - 1].value) {
        seq[i - 1] = FrameInput::S;
        return;
      }
    }
  }
  static void AddStopNNB_(FrameSequence& seq) {
    for (auto& i : seq) {
      if (i == FrameInput::S) return;
      if (!i.value) {
        i = FrameInput::S;
        return;
      }
    }
  }
  static void AddPushdown_(FrameSequence& seq) {
    if (seq.size() >= 5) {
      for (size_t i = seq.size() - 5; i < seq.size(); i++) seq[i] = FrameInput::D;
    }
  }

  void ResizeFrameSeq_(FrameSequence& seq, int row) const {
    size_t frame = move_search::GetFirstFrameOnRow(row + 1, LevelSpeed());
    seq.resize(frame);
  }

  void SequencePostProcess_(FrameSequence& seq, const Position& pos) const {
    ResizeFrameSeq_(seq, pos.x);
    int clear_lines = GetClearLines_(pos);
    if (ShouldPushdown_(lines_, clear_lines)) AddPushdown_(seq);
    if (cur_nnb_) AddStopNNB_(seq);
    if (ShouldNNB_(lines_, clear_lines)) AddStartNNB_(seq);
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
    SetAgentMode_();
    CalculateMoves_(true);
    cur_nnb_ = false;
    consecutive_fail_ = 0;
    run_score_ = 0;
    run_lines_ = 0;
    run_pieces_ = 0;
  }

  bool IsNoAdjMove(const Position& pos) const {
    return move_map_[pos.r][pos.x][pos.y] == kNoAdj;
  }
  bool IsAdjMove(const Position& pos) const {
    return move_map_[pos.r][pos.x][pos.y] == kHasAdj;
  }

  void DirectPlacement(const Position& pos, int next_piece) {
    if (game_over_) throw std::logic_error("already game over");
    if (next_piece < 0 || next_piece >= (int)kPieces) throw std::range_error("Invalid piece");
    uint8_t location = move_map_[pos.r][pos.x][pos.y];
    if (!(location == kNoAdj || std::any_of(moves_.adj.begin(), moves_.adj.end(), [&pos](auto i) {
            return std::find(i.second.begin(), i.second.end(), pos) != i.second.end();
          }))) {
      game_over_ = true;
      return;
    }
    if (StepGame_(pos, next_piece).first == -1) {
      game_over_ = true;
      return;
    }
  }

  // (score, lines)
  // score == -1 if invalid
  std::pair<int, int> InputPlacement(const Position& pos, int next_piece) {
    if (game_over_) throw std::logic_error("already game over");
    if (next_piece < 0 || next_piece >= (int)kPieces) throw std::range_error("Invalid piece");
    uint8_t location = move_map_[pos.r][pos.x][pos.y];
    if (!location) {
      consecutive_fail_++;
      return {-1, 0};
    }
    if (location == kNoAdj) {
      return StepGame_(pos, next_piece);
    } else {
      for (size_t i = 0; i < moves_.adj.size(); i++) {
        if (moves_.adj[i].first == pos) {
          initial_move_ = i;
          break;
        }
      }
      is_adj_ = true;
      CalculateMoves_(false);
      consecutive_fail_ = 0;
      return {0, 0};
    }
  }

  FrameSequence GetSequence(const Position& pos, bool is_final) const {
    auto seq = GetFrameSequenceStart<TAP_SPEED>(board_, LevelSpeed(), now_piece_, GetAdjDelay_(), pos);
    if (is_final) SequencePostProcess_(seq, pos);
    return seq;
  }

  std::pair<Position, FrameSequence> GetAdjPremove(const Position pos[7]) const {
    auto [idx, seq] = GetBestAdj<TAP_SPEED>(board_, LevelSpeed(), now_piece_, moves_, GetAdjDelay_(), pos);
    if (cur_nnb_) AddStopNNB_(seq);
    return {moves_.adj[idx].first, seq};
  }

  void FinishAdjSequence(FrameSequence& seq, const Position& intermediate_pos, const Position& final_pos, bool postprocess = true) const {
    GetFrameSequenceAdj<TAP_SPEED>(seq, board_, LevelSpeed(), now_piece_, intermediate_pos, final_pos);
    if (postprocess) SequencePostProcess_(seq, final_pos);
  }

  void SetNextPiece(int piece) {
    if (piece < 0 || piece >= (int)kPieces) throw std::range_error("Invalid piece");
    next_piece_ = piece;
  }

  const MoveMap& GetPossibleMoveMap() const { return move_map_; }
  const Board& GetBoard() const { return board_; }

  int GetRealLevel() const { return GetLevelByLines(lines_); }
  int GetStateLevel() const { return GetLevelByLines(GetStateLines()); }
  Level LevelSpeed() const { return GetLevelSpeedByLines(lines_); }
  bool IsAdj() const { return is_adj_; }
  int GetRealPieces() const { return pieces_; }
  int GetStatePieces() const { return pieces_ - (GetRealLines() - GetStateLines()) * 10 / 4; }
  int GetRealLines() const { return lines_; }
  int GetStateLines() const {
    int line_cap = (agent_mode_ == kNormalAgent ? 250 : 40) + (lines_ % 2);
    return std::min(lines_, line_cap);
  }
  int NowPiece() const { return now_piece_; }
  int NextPiece() const { return next_piece_; }
  bool IsOver() const { return game_over_ || consecutive_fail_ >= 1; }
  Position InitialMove() const {
    if (!is_adj_) throw std::logic_error("No initial move");
    return moves_.adj[initial_move_].first;
  }
  AgentMode GetAgentMode() const { return agent_mode_; }

  int RunPieces() const { return run_pieces_; }
  int RunLines() const { return run_lines_; }
  int RunScore() const { return run_score_; }
};
