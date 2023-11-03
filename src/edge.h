#ifndef EDGE_H_
#define EDGE_H_

#include <cstdint>
#include <vector>
#include <stdexcept>

#include "position.h"
#include "constexpr_helpers.h"

struct Edge {
  Position pos;
  std::vector<uint8_t> nxt;
};
struct NodeEdge {
  uint8_t count;
  // (next board ID, lines)
  std::vector<std::pair<int, int>> nexts;
  std::vector<Edge> edges;

  std::vector<uint8_t> GetBytes() const {
    int sz = nexts.size() * 5 + edges.size() * 4;
    for (auto& i : edges) sz += i.nxt.size();
    if (sz >= 65536 || nexts.size() >= 256 || edges.size() >= 256) throw std::out_of_range("output size too large");
    std::vector<uint8_t> ret(sz + 5);
    IntToBytes<uint16_t>(sz + 3, ret.data());
    ret[2] = count;
    ret[3] = nexts.size();
    int ind = 4;
    for (size_t i = 0; i < nexts.size(); i++) {
      if (nexts[i].second > 4) throw std::runtime_error("unexpected: invalid lines");
      IntToBytes<uint32_t>(nexts[i].first, ret.data() + ind);
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

  void ReadBytes(const uint8_t data[], bool with_header = false) {
    if (with_header) data += 2;
    count = *data++;
    nexts.resize(*data++);
    for (auto& i : nexts) {
      i.first = BytesToInt<uint32_t>(data);
      i.second = data[4];
      data += 5;
    }
    edges.resize(*data++);
    for (auto& ed : edges) {
      ed.pos.r = data[0];
      ed.pos.x = data[1];
      ed.pos.y = data[2];
      ed.nxt.resize(data[3]);
      data += 4;
      for (auto& i : ed.nxt) i = *data++;
    }
  }
};
using EdgeList = std::array<NodeEdge, 7>;

#endif // EDGE_H_
