#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <tsl/sparse_map.h>

#include "search.h"
#include "edge.h"

using BoardMap = tsl::sparse_map<Board, int, BoardHash>;

void Print(const Board& b, bool invert = true) {
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 10; j++) printf("%c ", ".1"[(b.Column(j) >> i & 1) ^ invert]);
    puts("");
  }
}

template <int R>
NodeEdge GetEdgeList(const Board& b, int piece, const PositionList<R>& pos_list, const BoardMap& boards) {
  std::bitset<R * 200> tot_bs;
  for (auto& x : pos_list) tot_bs |= x.second;
  if (tot_bs.count() >= 256) throw long(1);
  uint8_t mp[R * 200] = {};
  memset(mp, 0xff, sizeof(mp));
  NodeEdge ret;
  ret.nexts.reserve(tot_bs.count());
  ret.edges.reserve(pos_list.size());
  ret.count = b.Count();
  for (int i = tot_bs._Find_first(); i < (int)tot_bs.size(); i = tot_bs._Find_next(i)) {
    auto result = b.Place(piece, i / 200, i % 200 / 10, i % 10).ClearLines();
    int t = result.second.Count();
    if ((ret.count+4)%10 != t%10) throw short(1);
    auto it = boards.find(result.second);
    if (it != boards.end()) {
      mp[i] = ret.nexts.size();
      ret.nexts.push_back({it->second, result.first});
    }
  }
  for (auto &[pos, bs] : pos_list) {
    Edge ed = {pos, {}};
    ed.nxt.reserve(bs.count());
    for (int i = bs._Find_first(); i < (int)bs.size(); i = bs._Find_next(i)) {
      if (mp[i] != 0xff) ed.nxt.push_back(mp[i]);
    }
    if (ed.nxt.size()) {
      ed.nxt.shrink_to_fit();
      ret.edges.push_back(std::move(ed));
    }
  }
  ret.nexts.shrink_to_fit();
  ret.edges.shrink_to_fit();
  return ret;
}

std::filesystem::path BoardPath(const std::filesystem::path& pdir, int group) {
  return pdir / (std::to_string(group) + ".board");
}
std::filesystem::path EdgePath(const std::filesystem::path& pdir, int C, int group, const std::string& type) {
  return pdir / (std::to_string(C) + "." + std::to_string(group) + "." + type);
}

template <int C>
void Run(const std::filesystem::path pdir, const int group) {
  BoardMap nxt_map;
  {
    const int nxt_group = (group + 2) % 5;
    uint8_t buf[25];
    auto filename = BoardPath(pdir, nxt_group);
    int size = std::filesystem::file_size(filename) / 25;
    nxt_map.reserve(size);
    std::ifstream fin(filename);
    for (int i = 0; i < size; i++) {
      fin.read((char*)buf, sizeof(buf));
      nxt_map[Board::FromBytes(buf)] = i;
    }
  }
  int N = std::filesystem::file_size(BoardPath(pdir, group));
  std::ifstream fin(BoardPath(pdir, group));
  std::ofstream fout(EdgePath(pdir, C, group, "edges"));
  std::ofstream foffset(EdgePath(pdir, C, group, "offset"));
  std::ofstream flog(EdgePath(pdir, C, group, "edgelog"));

  uint8_t buf[25];
  char obuf[256];
  long long edc = 0, nxtc = 0, adjc = 0, c = 0, cc = 0, p = 0, pp = 0;
  auto start = std::chrono::steady_clock::now();
  auto prev = start;
  const int kLogInterval = 16384;
  // T, J, Z, O, S, L, I
  for (int id = 0; id < N; id++) {
    fin.read((char*)buf, sizeof(buf));
    Board board = Board::FromBytes(buf);
    EdgeList eds;
    eds[0] = GetEdgeList<4>(board, 0, SearchMoves<4, C, 5, 21>(board.TMap()), nxt_map);
    eds[1] = GetEdgeList<4>(board, 1, SearchMoves<4, C, 5, 21>(board.JMap()), nxt_map);
    eds[2] = GetEdgeList<2>(board, 2, SearchMoves<2, C, 5, 21>(board.ZMap()), nxt_map);
    eds[3] = GetEdgeList<1>(board, 3, SearchMoves<1, C, 5, 21>(board.OMap()), nxt_map);
    eds[4] = GetEdgeList<2>(board, 4, SearchMoves<2, C, 5, 21>(board.SMap()), nxt_map);
    eds[5] = GetEdgeList<4>(board, 5, SearchMoves<4, C, 5, 21>(board.LMap()), nxt_map);
    eds[6] = GetEdgeList<2>(board, 6, SearchMoves<2, C, 5, 21>(board.IMap()), nxt_map);
    uint64_t offset = fout.tellp();
    foffset.write((char*)&offset, sizeof(offset));
    bool flag = false;
    for (auto& i : eds) {
      edc += i.edges.size();
      nxtc += i.nexts.size();
      for (auto& j : i.edges) adjc += j.nxt.size();
      c++;
      if (i.nexts.size()) cc++, flag = true;

      auto buf = i.GetBytes();
      fout.write((char*)buf.data(), buf.size());
    }
    p++;
    if (flag) pp++;
    if (p % kLogInterval == 0) {
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> dur = end - start;
      std::chrono::duration<double> dur2 = end - prev;
      snprintf(obuf, sizeof(obuf), "%lld %lld %lld %lld %lld %lld %lld, %lf / %lf item/s\n",
          p, pp, c, cc, edc, nxtc, adjc, p / dur.count(), kLogInterval / dur2.count());
      flog << obuf << std::flush;
      prev = end;
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur = end - start;
  uint64_t offset = fout.tellp();
  foffset.write((char*)&offset, sizeof(offset));
  snprintf(obuf, sizeof(obuf), "%lld %lld %lld %lld %lld %lld %lld, %.3lf secs\n",
      p, pp, c, cc, edc, nxtc, adjc, dur.count());
  flog << obuf;
}

std::vector<std::vector<Board>> ReadGroups(const std::filesystem::path& pdir) {
  std::vector<std::vector<Board>> boards(5);
  uint8_t buf[25];
  while (fread(buf, 1, 25, stdin) == 25) {
    Board r = Board::FromBytes(buf);
    int cnt = r.Count();
    if (cnt & 1 || r.ClearLines().first != 0) continue;
    int group = cnt % 10 / 2;
    boards[group].push_back(r);
  }
  for (int i = 0; i < 5; i++) {
    auto& group = boards[i];
    std::sort(group.begin(), group.end(), [](const Board& x, const Board& y) {
        return x.Count() > y.Count(); });
    std::ofstream fout(BoardPath(pdir, i));
    for (auto& board : group) {
      board.ToBytes(buf);
      fout.write((char*)buf, sizeof(buf));
    }
  }
  return boards;
}

#include <sys/resource.h>

int GetCurrentMem() {
  FILE* fp = fopen("/proc/self/statm", "r");
  int x = 0;
  fscanf(fp, "%*d%d", &x);
  fclose(fp);
  return x;
}

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  std::filesystem::path pdir = argv[1];
  if (!std::filesystem::is_directory(pdir)) return 1;

  ReadGroups(pdir);
  Run<2>(pdir, 0);

  /*
  // Allocate small memory sections to prevent CoW on dict pages
  // Ensure RSS grows at least 128 MiB
  int start_mem = GetCurrentMem();
  while (true) {
    for (int j = 0; j < 131072; j++) *(new uint8_t) = 1;
    if (GetCurrentMem() - start_mem > 128 * 1024 / 4) break;
  }
  // exit(0) to prevent destructor accessing memory
  if (!fork()) {
    Run<1>(argv[1], boards.data());
    exit(0);
  }
  if (!fork()) {
    Run<2>(argv[2], boards.data());
    exit(0);
  }
  if (!fork()) {
    Run<3>(argv[3], boards.data());
    exit(0);
  }
  exit(0);
  */
}
