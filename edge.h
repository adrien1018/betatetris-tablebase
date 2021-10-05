#ifndef EDGE_H_
#define EDGE_H_

#include "position.h"

#include <vector>

struct Edge {
  Position pos;
  std::vector<uint8_t> nxt;
};
struct NodeEdge {
  uint8_t count;
  std::vector<std::pair<int, int>> nexts;
  std::vector<Edge> edges;

  std::vector<uint8_t> GetBytes() const {
    int sz = nexts.size() * 5 + edges.size() * 4;
    for (auto& i : edges) sz += i.nxt.size();
    if (sz >= 65536 || nexts.size() >= 256 || edges.size() >= 256) throw bool(1);
    std::vector<uint8_t> ret(sz + 5);
    *(uint16_t*)ret.data() = sz + 3;
    ret[2] = count;
    ret[3] = nexts.size();
    int ind = 4;
    for (size_t i = 0; i < nexts.size(); i++) {
      if (nexts[i].second > 4) throw char(1);
      *(uint32_t*)(ret.data() + ind) = nexts[i].first;
      ret[ind + 4] = nexts[i].second;
      ind += 5;
    }
    ret[ind++] = edges.size();
    for (size_t i = 0; i < edges.size(); i++) {
      ret[ind+0] = edges[i].pos.r;
      ret[ind+1] = edges[i].pos.x;
      ret[ind+2] = edges[i].pos.y;
      ret[ind+3] = edges[i].nxt.size();
      ind += 4;
      for (auto& j : edges[i].nxt) ret[ind++] = j;
    }
    if (ind != (int)ret.size()) throw int(1);
    return ret;
  }
};
using EdgeList = std::array<NodeEdge, 7>;

#endif // EDGE_H_
