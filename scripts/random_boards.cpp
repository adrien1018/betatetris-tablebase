#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "../board.h"

std::vector<std::vector<Board>> bs(101);

// sample some boards base on cell count
int main(int argc, char** argv) {
  if (argc < 2) return 1;

  {
    std::ifstream fin(argv[1]);
    CompactBoard b;
    while (fin.read((char*)b.data(), sizeof(b))) {
      Board nb(b);
      bs[nb.Count()/2].push_back(nb);
    }
  }
  for (int i = 0; i <= 100; i++) std::cout << i*2 << ' ' << bs[i].size() << std::endl;

  std::mt19937_64 gen;
  using mrand = std::uniform_int_distribution<int>;
  constexpr int P = 2;
  for (auto& i : bs) {
    if (i.empty()) continue;
    int ind[P];
    for (int j = 0; j < P; j++) ind[j] = mrand(0, i.size() - 1)(gen);
    std::sort(ind, ind + P);
    int x = std::unique(ind, ind + P) - ind;
    for (int j = 0; j < x; j++) {
      auto& b = i[ind[j]];
      std::cout << '{' << b.b1 << "u," << b.b2 << "u," << b.b3 << "u," << b.b4 << "u},\n";
    }
  }
}
