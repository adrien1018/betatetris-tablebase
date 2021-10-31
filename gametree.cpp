#pragma GCC target("avx")
#pragma GCC optimize("fast-math")

#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>

#include "edge.h"
#include "path.h"

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

// 0-based
constexpr int DropLevel(int lines) {
  if (lines < 130) return 2;
  if (lines < 230) return 1;
  return 0;
}

class GraphReader {
 private:
  std::filesystem::path pdir_;
  // always constant
  long total_[5];
  long count_[5][21];
  // constant wrt pieces
  // descending; sort by pieces asc (lines asc) so begin at lower speed (higher level pos)
  // index is 0-based (note: drop_level_ is also 0-based)
  long level_pos_[3];
  int pieces_, group_;
  // current state
  int drop_level_;
  long pos_;
  std::ifstream fin_, foffset_;

  void OpenGroup_(bool set_offset = true) {
    if (fin_.is_open()) fin_.close();
    if (foffset_.is_open()) foffset_.close();
    // filename is 1-based
    fin_.open(EdgePath(pdir_, drop_level_ + 1, group_, "edges"));
    foffset_.open(EdgePath(pdir_, drop_level_ + 1, group_, "offset"));
    if (set_offset) fin_.seekg(GetOffset_(pos_));
  }

  void SetPieces_(int pieces) {
    pos_ = 0;
    pieces_ = pieces;
    group_ = pieces * 4 % 10 / 2;
    for (int i = 20, sum = 0, prev_level = 2;; i--) {
      int lines = (pieces_ * 4 - (i * 10 + group_ * 2)) / 10;
      int cur_level = DropLevel(lines);
      if (i < 0) cur_level = 0; // fill remaining cells when i == -1
      for (; prev_level >= cur_level; prev_level--) level_pos_[prev_level] = sum;
      if (cur_level == 0) break; // also break when i == -1
      sum += count_[group_][i];
    }
    for (drop_level_ = 2; drop_level_ > 0 && !level_pos_[drop_level_ - 1]; drop_level_--);
    OpenGroup_(false);
  }

  uint64_t GetOffset_(long pos) {
    uint64_t ret;
    foffset_.seekg(pos * sizeof(ret));
    foffset_.read((char*)&ret, sizeof(ret));
    return ret;
  }
 public:
  GraphReader(const std::filesystem::path& pdir, int pieces = 0) : pdir_(pdir), total_{}, count_{} {
    foffset_.rdbuf()->pubsetbuf(nullptr, 0); // disable buffering on offset reading
    for (int group = 0; group < 5; group++) {
      std::ifstream fcount(CountPath(pdir_, group));
      long x, y;
      while (fcount >> x >> y) {
        count_[group][x / 10] = y;
        total_[group] += y;
      }
    }
    SetPieces_(pieces);
  }

  // also rewinds the stream
  void SetPieces(int pieces) {
    SetPieces_(pieces);
  }

  void Seek(long pos) {
    int drop_level = 2;
    while (drop_level > 0 && level_pos_[drop_level - 1] >= pos) drop_level--;
    pos_ = pos;
    if (drop_level_ != drop_level) {
      drop_level = drop_level_;
      OpenGroup_();
    } else {
      fin_.seekg(GetOffset_(pos_));
    }
  }

  long GetGroup() const { return group_; }
  long GetMaxSize() const { return *std::max_element(total_, total_ + 5); }
  long GetTotal() const { return total_[group_]; }
  long GetPos(int count) const {
    if (count < 0) return GetTotal();
    long ret = 0;
    for (int i = 20; i * 10 + group_ * 2 > count && i >= 0; i--) ret += count_[group_][i];
    return ret;
  }

  std::vector<uint8_t> ReadBatch(long maxsize) {
    long n = std::min(maxsize, total_[group_] - pos_);
    if (n <= 0) return {};
    std::vector<uint8_t> ret;
    while (true) {
      long rn = n;
      if (drop_level_ > 0) rn = std::min(rn, level_pos_[drop_level_ - 1] - pos_);
      size_t sz = GetOffset_(pos_ + rn) - fin_.tellg();
      size_t offset = ret.size();
      ret.resize(offset + sz);
      fin_.read((char*)(ret.data() + offset), sz);
      if (rn == n) break;
      n -= rn;
      pos_ += rn;
      drop_level_--;
      OpenGroup_();
    }
    pos_ += n;
    while (pos_ != total_[group_] && drop_level_ > 0 && level_pos_[drop_level_ - 1] == pos_) {
      drop_level_--;
      OpenGroup_();
    }
    return ret;
  }
};

constexpr int Level(int lines) {
  if (lines < 130) return 18;
  //if (lines >= 230) return 29;
  return lines / 10 + 6;
}
constexpr int Score(int lines, int level) {
  constexpr int kTable[] = {0, 40, 100, 300, 1200};
  return kTable[lines] * (level + 1);
}

std::vector<NodeValue> CalculateBatch(
    const std::vector<NodeValue>& prev, int pieces,
    const std::vector<uint8_t>& bytes, int n = 0) {
  EdgeList eds;
  std::vector<NodeValue> ret;
  ret.reserve(n);
  for (size_t offset = 0; offset < bytes.size();) {
    ret.emplace_back();
    for (int i = 0; i < 7; i++) {
      uint16_t sz = *(uint16_t*)(bytes.data() + offset);
      eds[i].ReadBytes(bytes.data() + offset + 2);
      offset += sz + sizeof(sz);
    }
    int count = eds[0].count;
    int lines = (pieces * 4 - count) / 10;
    int level = Level(lines);
    for (int t = 0; t < 7; t++) {
      auto& ed = eds[t];
      float cur = 0;
      for (auto& nxt : ed.edges) {
        NodeValue cur_arr{};
        for (auto& id : nxt.nxt) {
          cur_arr.MaxTo(prev[ed.nexts[id].first], Score(ed.nexts[id].second, level));
        }
        cur = std::max(cur, cur_arr.Dot(kTransitionProb[t]));
      }
      ret.back().val[t] = cur;
    }
  }
  return ret;
}

const int kMaxLines = 330;

void CalculatePiece(
    int piece, std::vector<NodeValue>& current,
    const std::vector<NodeValue>& prev, GraphReader& g) {
  g.SetPieces(piece);
  int total = g.GetTotal();
  int n = g.GetPos(piece * 4 - kMaxLines * 10);
  const int kBlockSize = 1024;
  for (int i = 0; i < n; i += kBlockSize) {
    int num = std::min(kBlockSize, n - i);
    auto arr = g.ReadBatch(num);
    auto ret = CalculateBatch(prev, piece, arr, num);
    memcpy(current.data() + i, ret.data(), num * sizeof(NodeValue));
  }
  if (n == total) {
    if (g.GetGroup() == 0) {
      for (int i = 0; i < 7; i++) printf("%.3f%c", current[total-1].val[i], " \n"[i==6]);
    }
  } else {
    memset(current.data() + n, 0, (total - n) * sizeof(NodeValue));
  }
}

#include <chrono>
int main(int argc, char** argv) {
  if (argc < 2) return 1;
  GraphReader g(argv[1]);
  int max_size = g.GetMaxSize();
  int max_pieces = (kMaxLines * 10 + 200) / 4;

  std::vector<NodeValue> current(max_size), prev(max_size, NodeValue{});
  auto start = std::chrono::steady_clock::now();
  for (int piece = max_pieces; piece >= 0; piece--) {
    CalculatePiece(piece, current, prev, g);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = end - start;
    printf("%d %.3lfs\n", piece, dur.count());
    current.swap(prev);
  }
}
