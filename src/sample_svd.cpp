#include <random>
#include <fstream>
#include <stdexcept>
#include <Eigen/Dense>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop

#include "game.h"
#include "files.h"
#include "evaluate.h"
#include "board_set.h"

namespace {

constexpr int kBucketSize = 32;
constexpr int kMaximum = 2400000;

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
  const int group = GetGroupByPieces(pieces);
  spdlog::info("Writing samples of group {}", group);
  const auto fname = SVDSamplePath(group);
  MkdirForFile(fname);
  std::ofstream fout(fname);
  if (!fout.is_open()) throw std::runtime_error("file open failed");
  fout.write(reinterpret_cast<const char*>(mask.data()), mask.size());
}

constexpr size_t kSVDColumns = (kLineCap + 1) / 2;
using MatrixSVD = Eigen::Matrix<float, Eigen::Dynamic, kSVDColumns>;
using MatrixSVDTrans = Eigen::Matrix<float, kSVDColumns, Eigen::Dynamic>;

void ReadGroup(int group, bool is_ev, std::vector<std::array<float, kSVDColumns>>& mat) {
  spdlog::info("Start reading {} samples of group {}", is_ev ? "ev" : "var", group);
  const auto fname_cells = SVDSampleCountPath(group);
  const size_t samples = std::filesystem::file_size(fname_cells);
  std::vector<uint8_t> cells(samples);
  {
    std::ifstream fin(fname_cells);
    fin.read(reinterpret_cast<char*>(cells.data()), cells.size());
    if ((size_t)fin.gcount() != cells.size()) throw std::runtime_error("sample file incorrect");
  }
  const size_t max_cells = *std::max_element(cells.begin(), cells.end());
  std::vector<ClassReader<BasicIOType<float>>> readers;
  const size_t max_pieces = ((kLineCap - 1) * 10 + max_cells) / 4;
  const size_t start_pieces = (group + (group & 1) * 5) / 2;
  for (size_t pieces = start_pieces; pieces <= max_pieces; pieces += 5) {
    readers.emplace_back(is_ev ? SVDEvPath(pieces) : SVDVarPath(pieces));
  }
  const size_t old_size = mat.size();
  mat.resize(old_size + samples);
  std::array<float, kSVDColumns>* ret = mat.data() + old_size;
  float buf[(kLineCap + 21) / 2] = {};
  for (size_t i = 0; i < samples; i++) {
    for (size_t j = 0; j < readers.size(); j++) buf[j] = readers[j].ReadOne();
    const size_t offset = (cells[i] - start_pieces * 4 + 19) / 20;
    for (size_t j = 0; j < kSVDColumns; j++) ret[i][j] = buf[j + offset];
  }
}

template <class Mat, class Stream>
void OutputMatrix(const Mat& mat, Stream& fout) {
  if (mat.cols() == 1) {
    for (long i = 0; i < mat.rows(); i++) {
      fout << fmt::format("{}", mat(i, 0)) << " \n"[i == mat.rows() - 1];
    }
    return;
  }
  for (long i = 0; i < mat.rows(); i++) {
    for (long j = 0; j < mat.cols(); j++) {
      fout << fmt::format("{}", mat(i, j)) << " \n"[j == mat.cols() - 1];
    }
  }
}

void OutputStats(const MatrixSVD& original, const MatrixSVD& reconstruct,
                 std::ofstream& fout, std::ofstream* fsample = nullptr) {
  float mse = std::sqrt((original - reconstruct).array().square().mean());
  Eigen::VectorXf row_max = (original - reconstruct).cwiseAbs().rowwise().maxCoeff();
  float mse_row = std::sqrt(row_max.array().square().mean()), extreme = 0;
  std::vector<float> percentile(11), high_percentile(10);
  {
    std::vector<float> data(row_max.data(), row_max.data() + original.rows());
    std::sort(data.begin(), data.end());
    for (size_t i = 0; i <= 10; i++) {
      percentile[i] = data[(data.size() - 1) * i / 10];
    }
    for (size_t i = 0; i < 10; i++) {
      high_percentile[i] = data[(data.size() - 1) * (i + 90) / 100];
    }
    extreme = data[(data.size() - 1) * 399999 / 400000];
  }
  {
    std::string str = fmt::format("MSE: {}; Rowwise MSE: {}", mse, mse_row);
    fout << str << '\n';
    spdlog::debug(str);
    str = fmt::format("Percentiles: {}", percentile);
    fout << str << '\n';
    spdlog::debug(str);
    str = fmt::format("90+ Percentiles: {}", high_percentile);
    fout << str << '\n';
    spdlog::debug(str);
  }
  fout << "Extreme:";
  for (long i = 0; i < original.rows(); i++) {
    if (row_max(i) >= extreme) {
      OutputMatrix(original.row(i), fout);
      OutputMatrix(reconstruct.row(i), fout);
      fout << "---\n";
    }
  }
  if (fsample) {
    for (long i = 0; i < original.rows(); i++) {
      *fsample << original(i, 0) << ' ' << row_max(i) << '\n';
    }
  }
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

void DoSVD(bool is_ev, float training_split, const std::vector<int>& ranks, size_t seed) {
  size_t total_samples = 0;
  for (int g = 0; g < kGroups; g++) total_samples += std::filesystem::file_size(SVDSampleCountPath(g));
  training_split = std::min(1.0f, std::max(0.0f, training_split));
  const size_t training_samples = total_samples * training_split;
  const size_t testing_samples = total_samples - training_samples;
  float max_val = 0;
  MatrixSVD training_mat(training_samples, kSVDColumns), testing_mat(testing_samples, kSVDColumns);
  {
    std::vector<std::array<float, kSVDColumns>> tmp_mat;
    tmp_mat.reserve(total_samples);
    for (int g = 0; g < kGroups; g++) ReadGroup(g, is_ev, tmp_mat);
    if (!is_ev) { // do SVD on stdev instead of variance
      for (auto& i : tmp_mat) {
        for (auto& j : i) j = std::sqrt(j);
      }
    }
    std::vector<size_t> perm(total_samples);
    for (size_t i = 0; i < total_samples; i++) perm[i] = i;
    std::mt19937_64 gen(seed);
    std::shuffle(perm.begin(), perm.end(), gen);
    for (size_t i = 0; i < training_samples; i++) {
      max_val = std::max(max_val, tmp_mat[perm[i]][0]);
      for (size_t j = 0; j < kSVDColumns; j++) training_mat(i, j) = tmp_mat[perm[i]][j];
    }
    for (size_t i = 0; i < testing_samples; i++) {
      for (size_t j = 0; j < kSVDColumns; j++) testing_mat(i, j) = tmp_mat[perm[i + training_samples]][j];
    }
  }
  /*
  MatrixSVD svd_mat = training_mat;
  for (size_t i = 0; i < training_samples; i++) {
    svd_mat.row(i) *= std::sqrt((training_mat(i, 0) + 1) / max_val);
  }
  */

  using namespace Eigen;

  spdlog::info("Start calculating SVD");
  JacobiSVD<Eigen::MatrixXf> svd(training_mat, ComputeThinU | ComputeThinV);
  Eigen::Matrix<float, kSVDColumns, 1> scale;
  MatrixSVDTrans scaled_v = svd.matrixV();
  {
    /*
    JacobiSVD<MatrixXf> svd_lin(scaled_v, ComputeThinU | ComputeThinV);
    scale = svd_lin.solve(training_mat.transpose()).array().square().rowwise().mean().sqrt();
    */
    MatrixXf u = svd.matrixU();
    auto sing = svd.singularValues();
    for (size_t i = 0; i < kSVDColumns; i++) u.col(i) *= sing(i);
    scale = u.array().square().colwise().mean().sqrt().transpose();
  }
  for (size_t i = 0; i < kSVDColumns; i++) scaled_v.col(i) *= scale(i);

  spdlog::info("SVD done, writing results");
  MkdirForFile(SVDResultPath(is_ev));
  std::ofstream fout(SVDResultPath(is_ev));
  fout << "Scaling factors:\n";
  OutputMatrix(scale, fout);
  fout << "V:\n";
  OutputMatrix(svd.matrixV(), fout);
  fout << "Scaled V:\n";
  OutputMatrix(scaled_v, fout);
  for (int rank : ranks) {
    spdlog::debug("Testing rank {}", rank);
    MatrixSVDTrans lin_mat = scaled_v.leftCols(rank);
    JacobiSVD<MatrixXf> svd_lin(lin_mat, ComputeThinU | ComputeThinV);
    {
      fout << "\nRank " << rank << "\nTraining\n";
      MatrixSVDTrans trans = training_mat.transpose();
      auto result = svd_lin.solve(trans);
      MatrixSVD reconstruct = result.transpose() * lin_mat.transpose();
      OutputStats(training_mat, reconstruct, fout);
    }
    if (testing_samples) {
      std::ofstream fsample(SVDResultListPath(is_ev, rank));
      fout << "Testing\n";
      MatrixSVDTrans trans = testing_mat.transpose();
      auto result = svd_lin.solve(trans);
      MatrixSVD reconstruct = result.transpose() * lin_mat.transpose();
      OutputStats(testing_mat, reconstruct, fout, &fsample);
    }
  }
}
