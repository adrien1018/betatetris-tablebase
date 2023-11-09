#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

#include "position.h"
#include "constexpr_helpers.h"

struct EvaluateNodeEdges {
  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;

  // edge for`evaluating; no position information available
  uint8_t cell_count;
  // (next board ID, lines)
  std::vector<std::pair<uint64_t, uint8_t>> next_ids;
  // indices of next_ids
  std::vector<uint8_t> non_adj;
  std::vector<std::vector<uint8_t>> adj;

  bool operator==(const EvaluateNodeEdges&) const = default;
  bool operator!=(const EvaluateNodeEdges&) const = default;

  size_t NumBytes() const {
    int sz = 3 + next_ids.size() * 5 + non_adj.size();
    for (auto& i : adj) {
      sz += i.size() + 1;
      if (i.size() >= 256) throw std::out_of_range("output size too large");
    }
    if (non_adj.size() >= 256 || adj.size() >= 256) throw std::out_of_range("output size too large");
    return sz;
  }

  void GetBytes(uint8_t ret[]) const {
    int sz = NumBytes();
    ret[0] = cell_count;
    // next ids
    int ind = 1;
    ret[ind++] = next_ids.size();
    for (auto& i : next_ids) {
      if (i.first >= (1ll << 32)) throw std::out_of_range("too many boards");
      IntToBytes<uint32_t>(i.first, &ret[ind]);
      ret[ind+4] = i.second;
      ind += 5;
    }
    // non_adjs
    ret[ind++] = non_adj.size();
    memcpy(&ret[ind], non_adj.data(), non_adj.size());
    ind += non_adj.size();
    // adjs
    ret[ind++] = adj.size();
    for (auto& adj_item : adj) {
      ret[ind++] = adj_item.size();
      memcpy(&ret[ind], adj_item.data(), adj_item.size());
      ind += adj_item.size();
    }
    if (ind != sz) throw std::runtime_error("size not match");
  }

  static EvaluateNodeEdges FromBytes(const uint8_t data[], size_t sz) {
    EvaluateNodeEdges ret;
    int ind = 0;
    ret.cell_count = data[ind++];
    ret.next_ids.resize(data[ind++]);
    for (auto& i : ret.next_ids) {
      i.first = BytesToInt<uint32_t>(data + ind);
      i.second = data[ind+4];
      ind += 5;
    }
    ret.non_adj.resize(data[ind++]);
    memcpy(ret.non_adj.data(), data, ret.non_adj.size());
    ind += ret.non_adj.size();
    ret.adj.resize(data[ind++]);
    for (auto& adj_item : ret.adj) {
      adj_item.resize(data[ind++]);
      memcpy(adj_item.data(), data, adj_item.size());
      ind += adj_item.size();
    }
    if (ind != (int)sz) throw std::runtime_error("size not match");
  }
};
