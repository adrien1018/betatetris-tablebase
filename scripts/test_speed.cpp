#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#define TESTING 1
#include "../src/move_search.h"
#include "../src/utils.h"
//#include "src/old/search.h"

constexpr Level kLevel = kLevel18;

constexpr int kFramesPerDrop = kLevel == kLevel18 ? 3 : kLevel == kLevel19 ? 2 : 1;
constexpr int kV = 2; // frames per tap
using TapSpeed = Tap30Hz;

constexpr int N = 16384;

Board boards[N];

__attribute__((optimize("O1"))) int main() {
  using namespace std::chrono;
  time_point<steady_clock> start;
  duration<double> diff;

  if (false) {
    using namespace Eigen;
    auto A = MatrixXf::Random(10000, 215); // about 1s / 10k rows
    start = steady_clock::now();
    JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
    diff = steady_clock::now() - start;
    std::cout << "SVD time: " << diff.count() << std::endl;

    Matrix<float, Dynamic, 32> linA = svd.matrixV().leftCols<32>();
    auto sing = svd.singularValues();
    for (size_t i = 0; i < 32; i++) linA.col(i) *= sing(i);
    JacobiSVD<MatrixXf> svd_lin(linA, ComputeThinU | ComputeThinV);
    auto B = MatrixXf::Random(150000, 215); // about 1s / 150k rows

    {
      start = steady_clock::now();
      auto Btrans = B.transpose();
      auto result = svd_lin.solve(Btrans);
      float p = result.sum();
      size_t x = result.rows();
      size_t y = result.cols();
      diff = steady_clock::now() - start;
      std::cout << "Solve time: " << diff.count() << ' ' << p << ' ' << x << ' ' << y << std::endl;
    }
  }

  if (true) {
    int cnt = 0, cnt2 = 0;

    for (int i = 0; i < N; i++) {
      uint8_t buf[25];
      for (int j = 0; j < 24; j++) buf[j] = 0xff;
      buf[23] = i >> 8;
      buf[24] = i;
      boards[i] = Board(buf);
    }

    for (int iter = 0; iter < 5; iter++) {
      start = steady_clock::now();
      for (int i = 0; i < N; i++) {
        auto a = MoveSearch<kLevel, 4, 61, TapSpeed>(boards[i].TMap());
        cnt += a.adj.size() + a.non_adj.size();
        cnt2 += a.non_adj.size();
        for (auto& x : a.adj) cnt2 += x.second.size();
      }
      diff = steady_clock::now() - start;
      std::cout << "New: " << N << " no-adj searches in " << diff.count() << "s (" << diff.count() / N * 1e6 << "us/search)\n";
    }

    for (int iter = 0; iter < 5; iter++) {
      start = steady_clock::now();
      for (int i = 0; i < N; i++) {
        auto a = MoveSearch<kLevel, 4, 18, TapSpeed>(boards[i].TMap());
        cnt += a.adj.size() + a.non_adj.size();
        cnt2 += a.non_adj.size();
        for (auto& x : a.adj) cnt2 += x.second.size();
      }
      diff = steady_clock::now() - start;
      std::cout << "New: " << N << " with-adj searches in " << diff.count() << "s (" << diff.count() / N * 1e6 << "us/search)\n";
    }

    TapSpeed taps_obj;
    for (int iter = 0; iter < 5; iter++) {
      PrecomputedTable table(kLevel, 4, 61, taps_obj.data());
      start = steady_clock::now();
      for (int i = 0; i < N; i++) {
        auto a = MoveSearch<4>(kLevel, 61, taps_obj.data(), table, boards[i].TMap());
        cnt += a.adj.size() + a.non_adj.size();
        cnt2 += a.non_adj.size();
        for (auto& x : a.adj) cnt2 += x.second.size();
      }
      diff = steady_clock::now() - start;
      std::cout << "New: " << N << " no-adj searches in " << diff.count() << "s (" << diff.count() / N * 1e6 << "us/search)\n";
    }

    for (int iter = 0; iter < 5; iter++) {
      PrecomputedTable table(kLevel, 4, 18, taps_obj.data());
      start = steady_clock::now();
      for (int i = 0; i < N; i++) {
        auto a = MoveSearch<4>(kLevel, 18, taps_obj.data(), table, boards[i].TMap());
        cnt += a.adj.size() + a.non_adj.size();
        cnt2 += a.non_adj.size();
        for (auto& x : a.adj) cnt2 += x.second.size();
      }
      diff = steady_clock::now() - start;
      std::cout << "New: " << N << " with-adj searches in " << diff.count() << "s (" << diff.count() / N * 1e6 << "us/search)\n";
    }

    /*
    cnt = cnt2 = 0;
    start = steady_clock::now();
    for (int i = 0; i < N; i++) {
      auto a = SearchMoves<4, kFramesPerDrop, kV, kAdjDelay>(boards[i].TMap());
      cnt += a.size();
      for (auto& x : a) cnt2 += x.second.count();
    }
    diff = steady_clock::now() - start;
    std::cout << "Old: " << N << " searches in " << diff.count() << "s (" << diff.count() / N * 1e6 << "us/search)\n";
    */
  }
}
