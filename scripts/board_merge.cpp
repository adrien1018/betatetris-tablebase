#include <string>
#include <fstream>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <tsl/sparse_set.h>
#pragma GCC diagnostic pop
#include "../src/board.h"

using BoardSet = tsl::sparse_set<CompactBoard, std::hash<CompactBoard>, std::equal_to<CompactBoard>,
      std::allocator<CompactBoard>, tsl::sh::power_of_two_growth_policy<2>,
      tsl::sh::exception_safety::basic, tsl::sh::sparsity::high>;

// sample some boards base on cell count
int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "Usage: ./board_merge [out_file] [partition] [tot_partition_pow] [mirror(0/1)] [reserve] [files...]\n";
    return 1;
  }
  const std::string out_file = argv[1];
  const int partition = std::stoi(argv[2]);
  const int total_partition_pow = std::stoi(argv[3]);
  const bool mirror = std::stoi(argv[4]);
  const size_t reserve = std::stoul(argv[5]);

  BoardSet st;
  st.reserve(reserve);
  auto Insert = [&](const CompactBoard& b) {
    auto hash = std::hash<CompactBoard>()(b) >> 7;
    if (int(hash & ((1 << total_partition_pow) - 1)) == partition) st.insert(b);
  };
  for (int i = 6; i < argc; i++) {
    size_t cnt = 0;
    std::ifstream fin(argv[i]);
    CompactBoard b;
    while (fin.read((char*)b.data(), sizeof(b))) {
      cnt++;
      Insert(b);
      if (false) {
        Board nb(b);
        uint32_t col0 = nb.Column(0);
        int left_height = nb.ColumnHeights()[0];
        if (__builtin_popcount(col0 + 1) == 1 && left_height < 16) {
          Board sb(nb.b1 & ~(15 << (16 - left_height)), nb.b2, nb.b3, nb.b4);
          auto fin = sb.ClearLines();
          if (fin.first == 0 || fin.first == 4) Insert(fin.second.ToBytes());
        }
      }
      if (mirror) {
        Board nb(b);
        uint64_t v1 = nb.Column(9) | (uint64_t)nb.Column(8) << 22 | (uint64_t)nb.Column(7) << 44;
        uint64_t v2 = nb.Column(6) | (uint64_t)nb.Column(5) << 22 | (uint64_t)nb.Column(4) << 44;
        uint64_t v3 = nb.Column(3) | (uint64_t)nb.Column(2) << 22 | (uint64_t)nb.Column(1) << 44;
        uint64_t v4 = nb.Column(0);
        Insert(Board(v1, v2, v3, v4).ToBytes());
      }
    }
    std::cout << cnt << " boards in " << argv[i] << ", current total " << st.size() << std::endl;
  }

  std::ofstream fout(out_file);
  for (auto& i : st) {
    fout.write((char*)i.data(), sizeof(i));
  }
}
