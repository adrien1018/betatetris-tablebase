#include <queue>
#include <random>
#include <spdlog/spdlog.h>
#include "game.h"
#include "files.h"
#include "evaluate.h"
#include "board_set.h"
#include "sample_svd.h"

namespace {

struct SampleItem {
  float mx;
  uint32_t idx;
  uint8_t cnt;
  bool operator<(const SampleItem& x) const {
    return mx > x.mx;
  }
};

void SampleZeros(const std::vector<NodeEval>& val, size_t num_samples, float high_ratio, size_t seed,
                 std::vector<uint8_t>& mask) {
  if (val.size() >= (1ll << 32)) throw std::length_error("too large");
  float ev[8];
  size_t num_high_samples = high_ratio * num_samples;
  spdlog::debug("Sampling {} zeros", num_samples);
  std::priority_queue<SampleItem> pq;
  size_t queue_size = 0;
  for (size_t i = 0; i < val.size(); i++) {
    val[i].GetEv(ev);
    float mx = 0.0;
    uint8_t cnt = 0;
    for (size_t j = 0; j < 7; j++) {
      if (ev[j] == 0.0f && !(mask[i] >> j & 1)) cnt++;
      if (ev[j] > mx) mx = ev[j];
    }
    if (cnt) {
      pq.push({mx, (uint32_t)i, cnt});
      queue_size += cnt;
      while (queue_size >= num_high_samples + pq.top().cnt) {
        queue_size -= pq.top().cnt;
        pq.pop();
      }
    }
  }
  while (pq.size()) {
    auto item = pq.top();
    pq.pop();
    val[item.idx].GetEv(ev);
    for (size_t j = 0; j < 7; j++) {
      if (ev[j] == 0.0f && !(mask[item.idx] >> j & 1)) mask[item.idx] |= 1 << j;
    }
  }
  num_samples -= num_high_samples;
  spdlog::debug("High zeros complete, sampling {} random zeros", num_samples);
  std::mt19937_64 gen(seed);
  using mrand = std::uniform_int_distribution<size_t>;
  while (num_samples) {
    size_t idx = mrand(0, val.size() - 1)(gen);
    uint8_t piece = mrand(0, 6)(gen);
    val[idx].GetEv(ev);
    if (ev[piece] == 0.0f && !(mask[idx] >> piece & 1)) {
      num_samples--;
      mask[idx] |= 1 << piece;
    }
  }
}

} // namespace

void SampleTrainingBoards(
    const std::vector<int>& start_pieces_group, size_t num_samples,
    float zero_ratio, float zero_high_ratio, float smooth_pow, size_t seed,
    const std::filesystem::path& output_path) {
  std::mt19937_64 gen(seed);
  std::array<std::vector<std::pair<uint32_t, uint32_t>>, kGroups> samples;
  for (size_t i = 0; i < kGroups; i++) {
    samples[i].reserve(num_samples * start_pieces_group.size() * 1.05 + 100);
  }
  for (size_t lgroup = 0; lgroup < start_pieces_group.size(); lgroup++) {
    int start_pieces = start_pieces_group[lgroup];
    std::vector<NodeEval> values;
    try {
      values = ReadValues(start_pieces);
    } catch (std::length_error&) {
      spdlog::error("Length error on value file. Does the evaluate file exist?");
      return;
    }
    spdlog::info("Sampling from piece {}", start_pieces);
    {
      auto mask = SampleFromEval(values, (num_samples * (1. - zero_ratio)) + 1, smooth_pow, gen());
      SampleZeros(values, num_samples * zero_ratio + 1, zero_high_ratio, gen(), mask);
      SampleMaskToIdx(mask, samples[GetGroupByPieces(start_pieces)], lgroup << 3);
    }
    for (int i = 1; i < kGroups; i++) {
      int group = GetGroupByPieces(start_pieces - i);
      spdlog::info("Sampling from piece {}", start_pieces - i);
      values = CalculatePiece(start_pieces - i, values, GetBoardCountOffset(group));
      auto mask = SampleFromEval(values, (num_samples * (1. - zero_ratio)) + 1, smooth_pow, gen());
      SampleZeros(values, num_samples * zero_ratio + 1, zero_high_ratio, gen(), mask);
      SampleMaskToIdx(mask, samples[group], lgroup << 3);
    }
  }
  std::vector<std::array<uint8_t, sizeof(CompactBoard) + 1>> output;
  {
    size_t total_size = 0;
    for (auto& i : samples) total_size += i.size();
    output.reserve(total_size);
  }
  for (size_t i = 0; i < kGroups; i++) {
    auto fname = BoardPath(i);
    size_t num_boards = BoardCount(fname);
    ClassReader<CompactBoard> reader(fname);
    auto all_boards = reader.ReadBatch(num_boards);
    decltype(output)::value_type buf;
    for (auto& [v, mark] : samples[i]) {
      memcpy(buf.data(), all_boards[v].data(), sizeof(CompactBoard));
      buf[sizeof(CompactBoard)] = mark;
      output.push_back(buf);
    }
    samples[i].clear();
    samples[i].shrink_to_fit();
  }
  std::vector<size_t> order(output.size());
  for (size_t i = 0; i < output.size(); i++) order[i] = i;
  std::shuffle(order.begin(), order.end(), gen);
  std::ofstream fout(output_path);
  for (size_t i : order) {
    fout.write(reinterpret_cast<const char*>(output[order[i]].data()), sizeof(output[0]));
  }
}
