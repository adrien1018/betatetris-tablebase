#include <random>
#include <spdlog/spdlog.h>
#include "game.h"
#include "files.h"
#include "evaluate.h"
#include "board_set.h"
#include "sample_svd.h"

void SampleTrainingBoards(
    const std::vector<int>& start_pieces_group, size_t num_samples, float smooth_pow, size_t seed,
    const std::filesystem::path& output_path) {
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
    SampleFromEval(values, num_samples, smooth_pow, seed + lgroup, lgroup << 3, samples[GetGroupByPieces(start_pieces)]);
    for (int i = 1; i < kGroups; i++) {
      int group = GetGroupByPieces(start_pieces - i);
      spdlog::info("Sampling from piece {}", start_pieces - i);
      values = CalculatePiece(start_pieces - i, values, GetBoardCountOffset(group));
      SampleFromEval(values, num_samples, smooth_pow, seed + lgroup, lgroup << 3, samples[group]);
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
  std::shuffle(order.begin(), order.end(), std::mt19937_64(seed * 3));
  std::ofstream fout(output_path);
  for (size_t i : order) {
    fout.write(reinterpret_cast<const char*>(output[order[i]].data()), sizeof(output[0]));
  }
}
