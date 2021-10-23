#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
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

template <int R> NodeEdge Test(const BoardMap& boards) {
  return NodeEdge{};
}

template <int C>
void Run(const std::string& name, const BoardMap boards_mp[], const std::vector<Board> boards[]) {
  FILE *fp = fopen(name.c_str(), "wb"), *logp = fopen((name + ".log").c_str(), "w");
  if (!fp || !logp) return;

  int edc = 0, nxtc = 0, adjc = 0, c = 0, cc = 0, p = 0, pp = 0;
  auto start = std::chrono::steady_clock::now();
  auto prev = start;
  {
    for (int group = 0; group < 5; group++) {
      uint32_t boards_count = boards[group].size();
      fwrite(&boards_count, 4, 1, fp);
    }
    for (int group = 0; group < 5; group++) {
      int nxt_group = (group + 2) % 5;
      const auto& nxt_map = boards_mp[nxt_group];
      // T, J, Z, O, S, L, I
      for (auto& board : boards[group]) {
        EdgeList eds;
        eds[0] = GetEdgeList<4>(board, 0, SearchMoves<4, C, 5, 21>(board.TMap()), nxt_map);
        eds[1] = GetEdgeList<4>(board, 1, SearchMoves<4, C, 5, 21>(board.JMap()), nxt_map);
        eds[2] = GetEdgeList<2>(board, 2, SearchMoves<2, C, 5, 21>(board.ZMap()), nxt_map);
        eds[3] = GetEdgeList<1>(board, 3, SearchMoves<1, C, 5, 21>(board.OMap()), nxt_map);
        eds[4] = GetEdgeList<2>(board, 4, SearchMoves<2, C, 5, 21>(board.SMap()), nxt_map);
        eds[5] = GetEdgeList<4>(board, 5, SearchMoves<4, C, 5, 21>(board.LMap()), nxt_map);
        eds[6] = GetEdgeList<2>(board, 6, SearchMoves<2, C, 5, 21>(board.IMap()), nxt_map);
        bool flag = false;
        for (auto& i : eds) {
          edc += i.edges.size();
          nxtc += i.nexts.size();
          for (auto& j : i.edges) adjc += j.nxt.size();
          c++;
          if (i.nexts.size()) cc++, flag = true;

          auto buf = i.GetBytes();
          fwrite(buf.data(), 1, buf.size(), fp);
        }
        p++;
        if (flag) pp++;
        if (p % 16384 == 0) {
          auto end = std::chrono::steady_clock::now();
          std::chrono::duration<double> dur = end - start;
          std::chrono::duration<double> dur2 = end - prev;
          fprintf(logp, "%d %d %d %d %d %d %d, %lf / %lf item/s\n", p, pp, c, cc, edc, nxtc, adjc, p / dur.count(), 16384 / dur2.count());
          fflush(logp);
          prev = end;
        }
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur = end - start;
  fprintf(logp, "%d %d %d %d %d %d %d, %.3lf secs\n", p, pp, c, cc, edc, nxtc, adjc, dur.count());
  fclose(logp);
  fclose(fp);
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
  if (argc < 4) return 1;

  std::vector<BoardMap> boards_mp(5);
  std::vector<std::vector<Board>> boards(5);
  /*for (int i = 0; i < 5; i++) {
    boards_mp[i].reserve(56000000);
    boards[i].reserve(56000000);
  }*/
  {
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
      boards_mp[group][r] = boards[group].size();
      boards[group].push_back(r);
    }
  }
  // Allocate small memory sections to prevent CoW on dict pages
  // Ensure RSS grows at least 128 MiB
  int start_mem = GetCurrentMem();
  while (true) {
    for (int j = 0; j < 131072; j++) *(new uint8_t) = 1;
    if (GetCurrentMem() - start_mem > 128 * 1024 / 4) break;
  }
  // exit(0) to prevent destructor accessing memory
  if (!fork()) {
    Run<1>(argv[1], boards_mp.data(), boards.data());
    exit(0);
  }
  if (!fork()) {
    Run<2>(argv[2], boards_mp.data(), boards.data());
    exit(0);
  }
  if (!fork()) {
    Run<3>(argv[3], boards_mp.data(), boards.data());
    exit(0);
  }
  exit(0);
}
