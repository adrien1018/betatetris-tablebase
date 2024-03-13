#include "simulate.h"

#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include "play.h"
#include "config.h"
#include "thread_pool.hpp"

namespace {

class RNGGym {
  uint32_t seed_; // 16-bit ints are slow
  uint8_t cnt_;
  int prev_;
  int iters_;

  void Advance() {
    uint32_t bit = ((seed_ >> 1) ^ (seed_ >> 9)) & 1;
    seed_ = seed_ >> 1 | bit << 15;
  }
 public:
  RNGGym(uint64_t seed = 0) {
    Reset(seed);
  }

  void Reset(uint64_t seed) {
    prev_ = 7;
    seed_ = (seed >> 8) & 0xffff;
    cnt_ = seed & 0xff;
    iters_ = cnt_ >> 4;
    if (!iters_) iters_ = 16;
    iters_ += 3;
  }

  int Spawn() {
    constexpr int kTable[8] = {2, 7, 8, 10, 11, 14, 18, 0};
    for (int i = 0; i < iters_; i++) Advance();
    cnt_++;
    int ind = ((seed_ >> 8) + cnt_) & 7;
    if (ind == 7 || ind == prev_) {
      Advance();
      ind = ((seed_ >> 8 & 7) + kTable[prev_]) % 7;
    }
    prev_ = ind;
    return ind;
  }
};

class RNGNormal {
  std::mt19937_64 rng_;
  int prev;
 public:
  RNGNormal(uint64_t seed = 0) {
    Reset(seed);
  }

  void Reset(uint64_t seed) {
    rng_.seed(seed);
    prev = std::uniform_int_distribution<int>(0, 6)(rng_);
  }

  int Spawn() {
    prev = std::discrete_distribution<int>(
        kTransitionProbInt[prev], kTransitionProbInt[prev] + kPieces)(rng_);
    return prev;
  }
};

template <class RNG>
std::vector<SimulateResult> Simulate(const uint64_t seeds[], size_t num) {
  Tetris game;
  RNG rng;
  Play play;
  std::vector<SimulateResult> ret;
  const char kPieceNames[] = "TJZOSLI";
  for (size_t i = 0; i < num; i++) {
    int seed = seeds[i];
    rng.Reset(seed);
    int now_piece = rng.Spawn();
    int nxt_piece = rng.Spawn();
    game.Reset(Board::Ones, 0, now_piece, nxt_piece);

    SimulateResult cur;
    cur.seed = seed;
    cur.piece_seq += kPieceNames[now_piece];
    int prev_lines = 0;
    while (!game.IsOver()) {
      auto strats = play.GetStrat(game);
      if (strats[0] == Position::Invalid) break;
      cur.piece_seq += kPieceNames[nxt_piece];
      nxt_piece = rng.Spawn();
      game.DirectPlacement(strats[game.NextPiece()], nxt_piece);
      cur.lines_seq += '0' + (game.GetLines() - prev_lines);
      for (int i = 0; i < 3; i++) {
        if (prev_lines < kLevelSpeedLines[i+1] && kLevelSpeedLines[i+1] <= game.GetLines()) {
          cur.transitions[i] = game.RunScore();
        }
      }
      prev_lines = game.GetLines();
    }
    cur.end_height = game.GetBoard().Height();
    for (int i = 0; i < 3; i++) {
      if (kLevelSpeedLines[i+1] > game.GetLines()) {
        cur.transitions[i] = game.RunScore();
      }
    }
    cur.score = game.RunScore();
    cur.lines = game.GetLines();
    ret.push_back(std::move(cur));
  }
  return ret;
}

template <class RNG>
std::vector<SimulateResult> SimulateParallel(const std::vector<uint64_t>& seeds) {
  BS::thread_pool pool(kParallel);
  auto result = pool.parallelize_loop(0, seeds.size(), [&](int l, int r){
    return Simulate<RNG>(seeds.data() + l, r - l);
  }).get();
  std::vector<SimulateResult> ret;
  for (auto& i : result) {
    for (auto& j : i) ret.push_back(std::move(j));
  }
  return ret;
}

std::vector<uint64_t> InputSeed(std::basic_istream<char>* in) {
  uint64_t seed;
  std::vector<uint64_t> ret;
  while (*in >> seed) ret.push_back(seed);
  return ret;
}

void OutputResult(std::basic_ostream<char>* out, const std::vector<SimulateResult>& res) {
  for (auto& i : res) {
    *out << i.seed << ',';
    for (int j = 0; j < 3; j++) *out << i.transitions[j] << ',';
    *out << i.score << ',' << i.lines << ',' << i.end_height << ','
         << i.piece_seq << ',' << i.lines_seq << '\n';
  }
}

} // namespace

std::vector<SimulateResult> Simulate(const std::vector<uint64_t>& seeds, bool gym_rng) {
  if (gym_rng) {
    return SimulateParallel<RNGGym>(seeds);
  } else {
    return SimulateParallel<RNGNormal>(seeds);
  }
}

void OutputSimulate(const std::string& seed_file, const std::string& out_file, bool gym_rng) {
  std::vector<uint64_t> seeds;
  if (seed_file == "-") {
    seeds = InputSeed(&std::cin);
  } else {
    std::ifstream fin(seed_file);
    seeds = InputSeed(&fin);
  }
  auto res = Simulate(seeds, gym_rng);
  if (out_file == "-") {
    OutputResult(&std::cout, res);
  } else {
    std::ofstream fout(out_file);
    OutputResult(&fout, res);
  }
}
