#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <fstream>
#include <algorithm>
#include <tsl/sparse_map.h>

#include "search.h"
#include "edge.h"
#include "path.h"

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

#include "thread_pool.hpp"
#include <deque>
#include <execution>

BoardMap LoadNextMap(const std::filesystem::path pdir, const int group) {
  BoardMap nxt_map;
  const int nxt_group = (group + 2) % 5;
  uint8_t buf[kBoardBytes];
  auto filename = BoardPath(pdir, nxt_group);
  int size = BoardCount(filename);
  nxt_map.reserve(size);
  std::ifstream fin(filename);
  for (int i = 0; i < size; i++) {
    fin.read((char*)buf, sizeof(buf));
    nxt_map[Board::FromBytes(buf)] = i;
  }
  return nxt_map;
}

template <int C>
void Run(const std::filesystem::path pdir, const int group, const BoardMap& nxt_map) {
  int N = BoardCount(BoardPath(pdir, group));
  std::ifstream fin(BoardPath(pdir, group));
  std::ofstream fout(EdgePath(pdir, C, group, "edges"));
  std::ofstream foffset(EdgePath(pdir, C, group, "offset"));
  std::ofstream flog(EdgePath(pdir, C, group, "edgelog"));

  struct JobResult {
    int edc, nxtc, adjc, c, cc, p, pp;
    std::vector<std::vector<uint8_t>> vec;
  };
  auto Job = [&](const std::vector<Board>& boards) {
    JobResult ret{};
    EdgeList eds;
    for (auto& board : boards) {
      // T, J, Z, O, S, L, I
      eds[0] = GetEdgeList<4>(board, 0, SearchMoves<4, C, 5, 21>(board.TMap()), nxt_map);
      eds[1] = GetEdgeList<4>(board, 1, SearchMoves<4, C, 5, 21>(board.JMap()), nxt_map);
      eds[2] = GetEdgeList<2>(board, 2, SearchMoves<2, C, 5, 21>(board.ZMap()), nxt_map);
      eds[3] = GetEdgeList<1>(board, 3, SearchMoves<1, C, 5, 21>(board.OMap()), nxt_map);
      eds[4] = GetEdgeList<2>(board, 4, SearchMoves<2, C, 5, 21>(board.SMap()), nxt_map);
      eds[5] = GetEdgeList<4>(board, 5, SearchMoves<4, C, 5, 21>(board.LMap()), nxt_map);
      eds[6] = GetEdgeList<2>(board, 6, SearchMoves<2, C, 5, 21>(board.IMap()), nxt_map);

      bool flag = false;
      for (auto& i : eds) {
        ret.edc += i.edges.size();
        ret.nxtc += i.nexts.size();
        for (auto& j : i.edges) ret.adjc += j.nxt.size();
        ret.c++;
        if (i.nexts.size()) ret.cc++, flag = true;
        ret.vec.push_back(i.GetBytes());
      }
      ret.p++;
      if (flag) ret.pp++;
    }
    return ret;
  };

  uint8_t buf[kBoardBytes];
  char obuf[256];
  long long edc = 0, nxtc = 0, adjc = 0, c = 0, cc = 0, p = 0, pp = 0;
  auto start = std::chrono::steady_clock::now();
  auto prev = start;
  const int kLogInterval = 16384;
  const int kBlockSize = 512;
  std::deque<std::vector<Board>> job_queue;
  std::deque<std::future<JobResult>> result_queue;
  thread_pool pool(16);
  auto Output = [&](bool wait) {
    if (result_queue.empty()) return false;
    if (!wait) {
      if (result_queue.front().wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    }
    auto res = result_queue.front().get();
    job_queue.pop_front();
    result_queue.pop_front();
    p += res.p, pp += res.pp, c += res.c, cc += res.cc, edc += res.edc, nxtc += res.nxtc, adjc += res.adjc;
    int cnt = 0;
    for (auto& i : res.vec) {
      uint64_t offset = fout.tellp();
      if (cnt++ == 0) foffset.write((char*)&offset, sizeof(offset));
      if (cnt == 7) cnt = 0;
      fout.write((char*)i.data(), i.size());
    }
    if (p % kLogInterval == 0) {
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> dur = end - start;
      std::chrono::duration<double> dur2 = end - prev;
      snprintf(obuf, sizeof(obuf), "%lld %lld %lld %lld %lld %lld %lld, %lf / %lf item/s\n",
          p, pp, c, cc, edc, nxtc, adjc, p / dur.count(), kLogInterval / dur2.count());
      flog << obuf << std::flush;
      prev = end;
    }
    return true;
  };
  for (int id = 0; id < N; id += kBlockSize) {
    while (result_queue.size() >= 160) Output(true);
    while (Output(false));
    std::vector<Board> boards;
    for (int i = id; i < std::min(N, id + kBlockSize); i++) {
      fin.read((char*)buf, sizeof(buf));
      boards.push_back(Board::FromBytes(buf));
    }
    job_queue.push_back(std::move(boards));
    result_queue.push_back(pool.submit(Job, job_queue.back()));
  }
  while (Output(true));
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
  uint8_t buf[kBoardBytes];
  while (fread(buf, 1, kBoardBytes, stdin) == kBoardBytes) {
    Board r = Board::FromBytes(buf);
    int cnt = r.Count();
    if (cnt & 1 || r.ClearLines().first != 0) continue;
    int group = cnt % 10 / 2;
    boards[group].push_back(r);
  }
  for (int i = 0; i < 5; i++) {
    auto& group = boards[i];
    std::vector<std::array<uint8_t, 26>> sort_key(group.size());
    std::vector<int> pos(group.size());
    for (size_t j = 0; j < group.size(); j++) {
      sort_key[j][0] = 200 - group[j].Count();
      group[j].ToBytes(sort_key[j].data() + 1);
      pos[j] = j;
    }
    std::sort(std::execution::par_unseq, pos.begin(), pos.end(),
        [&sort_key](int x, int y) { return sort_key[x] < sort_key[y]; });
    long long count[201] = {};
    std::ofstream fout(BoardPath(pdir, i));
    for (int j : pos) {
      auto& board = group[j];
      count[board.Count()]++;
      board.ToBytes(buf);
      fout.write((char*)buf, sizeof(buf));
    }
    std::ofstream fcount(CountPath(pdir, i));
    for (int i = 0; i < 201; i++) {
      if (count[i]) fcount << i << ' ' << count[i] << '\n';
    }
  }
  return boards;
}

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  std::filesystem::path pdir = argv[1];
  if (!std::filesystem::is_directory(pdir)) return 1;

  ReadGroups(pdir);
  for (int i = 0; i < 5; i++) {
    BoardMap nxt_map = LoadNextMap(pdir, i);
    Run<1>(pdir, i, nxt_map);
    Run<2>(pdir, i, nxt_map);
    Run<3>(pdir, i, nxt_map);
  }
}
