#pragma GCC target("avx")
#pragma GCC optimize("fast-math")

#include <cstdio>
#include <string>
#include <algorithm>

#include "edge.h"

struct alignas(32) TransitionProb {
  float val[8];
};

struct alignas(32) NodeValue {
  float val[8]; // the last one is dummy
  void MaxTo(const NodeValue& x, float m) {
    for (int i = 0; i < 8; i++) val[i] = std::max(val[i], x.val[i] + m);
  }
  float Dot(const TransitionProb x) const {
    float ret = 0;
    for (int i = 0; i < 8; i++) ret += val[i] * x.val[i];
    return ret;
  }
};

constexpr TransitionProb kTransitionProb[] = {
  {1./32, 5./32, 6./32, 5./32, 5./32, 5./32, 5./32}, // T
  {6./32, 1./32, 5./32, 5./32, 5./32, 5./32, 5./32}, // J
  {5./32, 6./32, 1./32, 5./32, 5./32, 5./32, 5./32}, // Z
  {5./32, 5./32, 5./32, 2./32, 5./32, 5./32, 5./32}, // O
  {5./32, 5./32, 5./32, 5./32, 2./32, 5./32, 5./32}, // S
  {6./32, 5./32, 5./32, 5./32, 5./32, 1./32, 5./32}, // L
  {5./32, 5./32, 5./32, 5./32, 6./32, 5./32, 1./32}, // I
};

class GraphReader {
 private:
  FILE* fp_;
  int pos_;
  std::vector<long> fpos_list_;

  bool ReadOne_(NodeEdge* nd = nullptr) const {
    uint16_t sz;
    if (!fread(&sz, 2, 1, fp_)) return false;
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, fp_);
    if (nd) nd->ReadBytes(buf.data());
    return true;
  }
 public:
  GraphReader(const std::string& filename) : pos_(0) {
    fp_ = fopen(filename.c_str(), "rb");
    if (!fp_) throw 0;

    fpos_list_.push_back(0);
    while (ReadOne_()) {
      for (int i = 0; i < 6; i++) ReadOne_();
      fpos_list_.push_back(ftell(fp_));
    }
  }
  ~GraphReader() { fclose(fp_); }

  void Rewind() { rewind(fp_); pos_ = 0; }
  int Position() const { return pos_; }
  int Count() const { return fpos_list_.size(); }
  void Seek(int pos) {
    fseek(fp_, fpos_list_[pos], SEEK_SET);
    pos_ = pos;
  }

  std::pair<bool, EdgeList> Read() {
    EdgeList ret;
    for (int i = 0; i < 7; i++) {
      if (!ReadOne_(&ret[i])) return {false, ret};
    }
    pos_++;
    return {true, ret};
  }
};

#include "board.h"
std::array<int, 5> GetSizes() {
  std::array<int, 5> ret{};
  uint8_t buf[25];
  while (fread(buf, 1, 25, stdin) == 25) {
    uint64_t cols[10] = {};
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 20; j++) {
        int x = j * 10 + i;
        cols[i] |= (uint64_t)(buf[x / 8] >> (x % 8) & 1) << j;
      }
    }
    Board r{cols[0] | cols[1] << 22 | cols[2] << 44,
            cols[3] | cols[4] << 22 | cols[5] << 44,
            cols[6] | cols[7] << 22 | cols[8] << 44,
            cols[9]};
    int cnt = r.Count();
    if (cnt & 1 || r.ClearLines().first != 0) continue;
    int group = cnt % 10 / 2;
    ret[group]++;
  }
  return ret;
}

constexpr int DropLevel(int lines) {
  if (lines < 130) return 2;
  if (lines < 230) return 1;
  return 0;
}
constexpr int Level(int lines) {
  if (lines < 130) return 18;
  //if (lines >= 230) return 29;
  return lines / 10 + 6;
}
constexpr int Score(int lines, int level) {
  constexpr int kTable[] = {0, 40, 100, 300, 1200};
  return kTable[lines] * (level + 1);
}

const int kMaxLines = 330;

#include <set>
void CalculatePiece(
    int piece, std::vector<NodeValue>& current, const std::vector<NodeValue>& prev,
    const std::array<int, 5>& sizes, GraphReader g[], std::set<int> to_print = {}) {
  static EdgeList eds[3];
  int group = piece * 4 % 10 / 2;
  int offset = 0;
  for (int i = 0; i < group; i++) offset += sizes[i];
  const int min_drop_level = DropLevel(piece * 4 / 10);
  const int max_drop_level = DropLevel(std::max(0, (piece * 4 - 200) / 10));
  printf("%d %d %d %d\n", group, piece*4/10, min_drop_level, max_drop_level);
  for (int l = min_drop_level; l <= max_drop_level; l++) g[l].Seek(offset);

  for (int i = 0; i < sizes[group]; i++) {
    for (int l = min_drop_level; l <= max_drop_level; l++) eds[l] = g[l].Read().second;
    int count = eds[min_drop_level][0].count;
    int lines = (piece * 4 - count) / 10;
    if (lines >= kMaxLines) {
      for (int t = 0; t < 8; t++) current[i].val[t] = 0;
      continue;
    }
    int drop_level = DropLevel(lines);
    int level = Level(lines);
    //bool cur_print = to_print.count(i);
    for (int t = 0; t < 7; t++) {
      auto& ed = eds[drop_level][t];
      //if (cur_print) printf("%d %d %d\n", t, (int)ed.edges.size(), (int)g[drop_level].Position());
      float cur = 0;
      for (auto& nxt : ed.edges) {
        NodeValue cur_arr{};
        //if (cur_print) printf("%d\n", (int)nxt.nxt.size());
        for (auto& id : nxt.nxt) {
          cur_arr.MaxTo(prev[ed.nexts[id].first], Score(ed.nexts[id].second, level));
          /*if (cur_print) {
            printf("%d %d ", ed.nexts[id].first, Score(ed.nexts[id].second, level));
            for (int z = 0; z < 7; z++) printf("%.3f%c", prev[ed.nexts[id].first].val[z], ", "[z == 6]);
            for (int z = 0; z < 7; z++) printf("%.3f%c", cur_arr.val[z], ",\n"[z == 6]);
          }*/
        }
        cur = std::max(cur, cur_arr.Dot(kTransitionProb[t]));
        //if (cur_print) printf("%.3lf %.3lf\n", cur, cur_arr.Dot(kTransitionProb[t]));
      }
      current[i].val[t] = cur;
    }
    if (count == 0) {
      printf("%d ", level);
      for (int t = 0; t < 7; t++) printf("%.3f%c", current[i].val[t], " \n"[t == 6]);
    }
  }
  if ((piece <= 30 && piece >= 20) || (piece && piece % 60 == 0)) {
    printf("1234567 %d %d\n", piece, sizes[group]);
    for (int i = 0; i < sizes[group]; i++) {
      for (int t = 0; t < 7; t++) printf("%.3f%c", current[i].val[t], " \n"[t == 6]);
    }
  }
  fflush(stdout);
}

#include <chrono>
int main(int argc, char** argv) {
  if (argc < 4) return 1;
  auto sizes = GetSizes(); //
  GraphReader g[] = {std::string(argv[1]), std::string(argv[2]), std::string(argv[3])};
  int max_size = *std::max_element(sizes.begin(), sizes.end());
  int max_pieces = (kMaxLines * 10 + 200) / 4;

  std::vector<NodeValue> current(max_size), prev(max_size, NodeValue{});
  auto start = std::chrono::steady_clock::now();
  for (int piece = max_pieces; piece >= 0; piece--) {
    CalculatePiece(piece, current, prev, sizes, g);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = end - start;
    printf("%d %.3lfs\n", piece, dur.count());
    current.swap(prev);
  }
}
