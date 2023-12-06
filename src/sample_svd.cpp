#pragma GCC optimize("fast-math")

#include <random>
#include <fstream>
#include <stdexcept>
#include <spdlog/spdlog.h>

#include "files.h"
#include "evaluate.h"
#include "board_set.h"

namespace {

static constexpr int kBucketSize = 32;
static constexpr int kMaximum = 2400000;

size_t GetBucket(float a) {
  return (size_t)a / kBucketSize;
}

std::vector<uint8_t> DoSample(const std::vector<NodeEval>& val, size_t num_samples, float smooth_pow, size_t seed) {
  if (val.size() >= (1ll << 32)) throw std::length_error("too large");
  std::vector<size_t> distribution(kMaximum / kBucketSize);
  size_t nonzeros = 0;
  for (auto& i : val) {
    float ev[8];
    i.GetEv(ev);
    for (size_t j = 0; j < 7; j++) {
      if (ev[j] != 0.0f) distribution[GetBucket(ev[j])]++, nonzeros++;
    }
  }
  auto GetTotal = [&](float multiplier) {
    size_t total = 0;
    for (auto& i : distribution) total += (size_t)std::ceil(std::pow(i, smooth_pow) * multiplier);
    return total;
  };
  float multiplier;
  {
    float l = 0, r = (float)num_samples / GetTotal(1);
    while (r - l >= 1e-6) {
      float m = (l + r) / 2;
      if (GetTotal(m) >= num_samples) {
        r = m;
      } else {
        l = m;
      }
    }
    multiplier = r;
  }
  multiplier = std::min(1.0f, multiplier);
  if (multiplier == 1.0f) {
    spdlog::warn("Multiplier is 1. This may cause unexpected low number of samples. Try increasing --pow.");
  }
  spdlog::debug("Multiplier determined: {}, samples = {}", multiplier, GetTotal(multiplier));

  std::vector<size_t> remaining(distribution.size());
  for (size_t i = 0; i < distribution.size(); i++) {
    remaining[i] = (size_t)std::ceil(std::pow(distribution[i], smooth_pow) * multiplier);
  }
  std::vector<uint8_t> result(val.size());
  std::mt19937_64 gen(seed);
  using mrand = std::uniform_int_distribution<size_t>;
  size_t finished = 0;
  while (finished * 8 < nonzeros - finished) {
    uint32_t v = mrand(0, val.size() - 1)(gen);
    uint32_t p = mrand(0, kPieces - 1)(gen);
    if (result[v] >> p & 1) continue;
    float ev[8];
    val[v].GetEv(ev);
    size_t bucket = GetBucket(ev[p]);
    if (ev[p] == 0.0f || !remaining[bucket]) continue;
    result[v] |= 1 << p;
    if (!--remaining[bucket]) finished += distribution[bucket];
  }

  std::vector<std::vector<std::pair<uint32_t, uint8_t>>> lst(distribution.size());
  for (size_t i = 0; i < distribution.size(); i++) {
    if (remaining[i]) lst[i].reserve(distribution[i]);
  }
  for (size_t i = 0; i < val.size(); i++) {
    float ev[8];
    val[i].GetEv(ev);
    for (size_t j = 0; j < 7; j++) {
      int bucket = GetBucket(ev[j]);
      if (ev[j] != 0.0f && !(result[i] >> j & 1) && remaining[bucket]) lst[bucket].push_back({i, j});
    }
  }
  for (size_t i = 0; i < distribution.size(); i++) {
    if (!remaining[i]) continue;
    if (remaining[i] * 3 < lst[i].size()) {
      for (size_t j = 0; j < remaining[i];) {
        size_t idx = mrand(0, lst[i].size() - 1)(gen);
        auto [v, p] = lst[i][idx];
        if (result[v] >> p & 1) continue;
        result[v] |= 1 << p;
        j++;
      }
    } else {
      std::shuffle(lst[i].begin(), lst[i].end(), gen);
      for (size_t j = 0; j < remaining[i]; j++) {
        auto [v, p] = lst[i][j];
        result[v] |= 1 << p;
      }
    }
  }
  result[0] = 0x7f; // always sample the first
  return result;
}

void WriteSample(int pieces, const std::vector<uint8_t>& mask) {
  int group = GetGroupByPieces(pieces);
  spdlog::info("Writing samples of group {}", group);
  auto fname = SVDSamplePath(group);
  MkdirForFile(fname);
  std::ofstream fout(fname);
  if (!fout.is_open()) throw std::runtime_error("file open failed");
  fout.write(reinterpret_cast<const char*>(mask.data()), mask.size());
}

} // namespace

void RunSample(int start_pieces, size_t num_samples, float smooth_pow, size_t seed) {
  if (start_pieces < kGroups) throw std::range_error("start_piece too small");
  if (smooth_pow > 1) {
    spdlog::warn("Exponent larger than 1. Setting to 1.");
    smooth_pow = 1;
  }
  spdlog::info("Start sampling about {} samples from each group", num_samples);
  std::vector<NodeEval> values;
  try {
    values = ReadValues(start_pieces);
  } catch (std::length_error&) {
    spdlog::error("Length error on value file. Does the evaluate file exist?");
    return;
  }
  spdlog::info("Sampling from piece {}", start_pieces);
  WriteSample(start_pieces, DoSample(values, num_samples, smooth_pow, seed));
  for (int i = 1; i < kGroups; i++) {
    spdlog::info("Sampling from piece {}", start_pieces - i);
    values = CalculatePiece(start_pieces - i, values, GetBoardCountOffset(GetGroupByPieces(start_pieces - i)));
    WriteSample(start_pieces - i, DoSample(values, num_samples, smooth_pow, seed + i));
  }
}
