#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

#include "position.h"
#include "constexpr_helpers.h"

class EvaluateNodeEdges {
  struct SubsetCalculator {
    const std::vector<std::vector<uint8_t>>& adj;
    std::vector<uint8_t>& adj_subset;
    std::vector<std::pair<uint8_t, int>>& subset_idx_prev; // (idx, prev)

    std::vector<std::vector<uint8_t>> contain_next;
    std::vector<uint8_t> adj_idx, remaining;
    std::vector<uint8_t> ref_cnt, selected;

    void RecursiveCalculate(size_t l, size_t r, int prev) {
      {
        std::vector<uint8_t> n_ref_cnt(ref_cnt.size());
        for (size_t i = l; i < r; i++) {
          for (auto& x : adj[adj_idx[i]]) n_ref_cnt[x]++;
        }
        for (size_t i = 0; i < ref_cnt.size(); i++) if (ref_cnt[i] != n_ref_cnt[i]) throw;
      }
      if (r - l == 1) {
        for (auto& x : adj[adj_idx[l]]) {
          if (selected[x]) continue;
          subset_idx_prev.push_back({x, prev});
          prev = subset_idx_prev.size() - 1;
        }
        adj_subset[adj_idx[l]] = prev;
        return;
      }
      std::vector<uint8_t> tmp_ref_cnt(ref_cnt);
      size_t finished = std::partition(adj_idx.begin() + l, adj_idx.begin() + r,
          [&](uint8_t idx) { return remaining[idx] > 0; }) - adj_idx.begin();
      for (size_t i = finished; i < r; i++) {
        adj_subset[adj_idx[i]] = prev;
        for (auto& x : adj[adj_idx[i]]) ref_cnt[x]--;
      }
      if (l == finished) {
        ref_cnt.swap(tmp_ref_cnt);
        return;
      }
      size_t split_id = 0;
      { // split by the most occuring next
        uint8_t mx = 0;
        for (size_t i = 0; i < ref_cnt.size(); i++) {
          if (selected[i]) continue;
          if (ref_cnt[i] > mx) mx = ref_cnt[i], split_id = i;
        }
      }
      size_t mid = std::partition(adj_idx.begin() + l, adj_idx.begin() + finished,
          [&](uint8_t idx) { return contain_next[idx][split_id]; }) - adj_idx.begin();
      if (l == mid) throw std::logic_error("what?");
      std::vector<uint8_t> n_ref_cnt(ref_cnt.size());
      for (size_t i = mid; i < finished; i++) {
        for (auto& x : adj[adj_idx[i]]) n_ref_cnt[x]++;
      }
      for (size_t i = 0; i < ref_cnt.size(); i++) ref_cnt[i] -= n_ref_cnt[i];
      // selected portion
      selected[split_id] = true;
      for (size_t i = l; i < mid; i++) remaining[adj_idx[i]]--;
      subset_idx_prev.push_back({(uint8_t)split_id, prev});
      RecursiveCalculate(l, mid, subset_idx_prev.size() - 1);
      selected[split_id] = false;
      for (size_t i = l; i < mid; i++) remaining[adj_idx[i]]++;
      // not selected portion
      if (mid != finished) {
        n_ref_cnt.swap(ref_cnt);
        RecursiveCalculate(mid, finished, prev);
      }
      ref_cnt.swap(tmp_ref_cnt);
    }

    SubsetCalculator(size_t next_sz,
                     const std::vector<std::vector<uint8_t>>& adj,
                     std::vector<uint8_t>& adj_subset,
                     std::vector<std::pair<uint8_t, int>>& subset_idx_prev) :
        adj(adj), adj_subset(adj_subset), subset_idx_prev(subset_idx_prev),
        contain_next(adj.size(), std::vector<uint8_t>(next_sz)),
        adj_idx(adj.size()), remaining(adj.size()),
        ref_cnt(next_sz), selected(next_sz) {
      adj_subset.resize(adj.size());
      for (size_t i = 0; i < adj.size(); i++) {
        adj_idx[i] = i;
        remaining[i] = adj[i].size();
        for (auto& x : adj[i]) ref_cnt[x]++, contain_next[i][x] = true;
      }
      RecursiveCalculate(0, adj.size(), -1);
    }
  };
 public:
  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;

  // edge for`evaluating; no position information available
  uint8_t cell_count;
  bool use_subset;
  // (next board ID, lines)
  std::vector<std::pair<uint64_t, uint8_t>> next_ids;
  // indices of next_ids
  std::vector<uint8_t> non_adj;
  std::vector<std::vector<uint8_t>> adj;
  // subset construction for faster calculation
  std::vector<uint8_t> adj_subset;
  std::vector<std::pair<uint8_t, int>> subset_idx_prev; // (idx, prev)

  EvaluateNodeEdges() : cell_count(), use_subset() {}

  bool operator==(const EvaluateNodeEdges&) const = default;
  bool operator!=(const EvaluateNodeEdges&) const = default;

  void CalculateSubset() {
    SubsetCalculator(next_ids.size(), adj, adj_subset, subset_idx_prev);
  }

  void CalculateAdj() {
    adj.clear();
    std::vector<std::vector<uint8_t>> lst(subset_idx_prev.size());
    for (size_t i = 0; i < subset_idx_prev.size(); i++) {
      auto& item = subset_idx_prev[i];
      if (item.second != -1) lst[i] = lst[item.second];
      lst[i].push_back(item.first);
    }
    // cannot use move, since might be reused
    for (auto i : adj_subset) adj.emplace_back(lst[i]);
  }

  size_t NumBytes() const {
    if (non_adj.size() >= 256) throw std::out_of_range("non_adj too large");
    size_t sz = 2 + 1 + next_ids.size() * 5 + 1 + non_adj.size();
    if (use_subset) {
      if (adj_subset.size() >= 256 || subset_idx_prev.size() >= 256) {
        throw std::out_of_range("subset too large");
      }
      sz += 1 + adj_subset.size() + 1 + subset_idx_prev.size() * 2;
    } else {
      if (adj.size() >= 256) throw std::out_of_range("adj too large");
      sz += 1;
      for (auto& i : adj) {
        sz += i.size() + 1;
        if (i.size() >= 256) throw std::out_of_range("adj[i] too large");
      }
    }
    return sz;
  }

  void GetBytes(uint8_t ret[]) const {
    size_t sz = NumBytes();
    ret[0] = cell_count;
    ret[1] = use_subset;
    // next ids
    size_t ind = 2;
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
    if (use_subset) {
      ret[ind++] = adj_subset.size();
      memcpy(&ret[ind], adj_subset.data(), adj_subset.size());
      ind += adj_subset.size();
      ret[ind++] = subset_idx_prev.size();
      for (auto& i : subset_idx_prev) {
        if (i.second >= 255 || i.second < -1) throw std::runtime_error("invalid prev");
        ret[ind] = i.first;
        ret[ind+1] = i.second == -1 ? 255 : i.second;
        //IntToBytes<uint16_t>(i.second == -1 ? 65536 : i.second, &ret[ind+1]);
        ind += 2;
      }
    } else {
      ret[ind++] = adj.size();
      for (auto& adj_item : adj) {
        ret[ind++] = adj_item.size();
        memcpy(&ret[ind], adj_item.data(), adj_item.size());
        ind += adj_item.size();
      }
    }
    if (ind != sz) throw std::runtime_error("size not match");
  }

  static EvaluateNodeEdges FromBytes(const uint8_t data[], size_t sz) {
    EvaluateNodeEdges ret;
    size_t ind = 0;
    ret.cell_count = data[ind++];
    ret.use_subset = data[ind++];
    ret.next_ids.resize(data[ind++]);
    for (auto& i : ret.next_ids) {
      i.first = BytesToInt<uint32_t>(data + ind);
      i.second = data[ind+4];
      ind += 5;
    }
    ret.non_adj.resize(data[ind++]);
    memcpy(ret.non_adj.data(), data + ind, ret.non_adj.size());
    ind += ret.non_adj.size();
    if (ret.use_subset) {
      ret.adj_subset.resize(data[ind++]);
      memcpy(ret.adj_subset.data(), data + ind, ret.adj_subset.size());
      ind += ret.adj_subset.size();
      ret.subset_idx_prev.resize(data[ind++]);
      for (auto& i : ret.subset_idx_prev) {
        i.first = data[ind];
        i.second = data[ind+1];
        if (i.second == 255) i.second = -1;
        //i.second = BytesToInt<uint16_t>(data + ind + 1);
        //if (i.second == 65536) i.second = -1;
        ind += 2;
      }
    } else {
      ret.adj.resize(data[ind++]);
      for (auto& adj_item : ret.adj) {
        adj_item.resize(data[ind++]);
        memcpy(adj_item.data(), data + ind, adj_item.size());
        ind += adj_item.size();
      }
    }
    if (ind != sz) throw std::runtime_error("size not match");
    return ret;
  }
};
