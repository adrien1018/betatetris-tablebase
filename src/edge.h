#pragma once

#include <cstdint>
#include <deque>
#include <vector>
#include <stdexcept>

#include "position.h"
#include "io_helpers.h"
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

    void Reorder() {
      std::vector<std::vector<std::pair<uint8_t, int>>> ed(subset_idx_prev.size() + 1);
      for (size_t i = 0; i < subset_idx_prev.size(); i++) {
        auto& item = subset_idx_prev[i];
        ed[item.second + 1].push_back({item.first, i});
      }
      std::vector<int> reverse_order(subset_idx_prev.size());
      std::deque<int> queue = {-1};
      size_t o1 = subset_idx_prev.size();
      subset_idx_prev.clear();
      while (queue.size()) {
        int cur = queue.front();
        queue.pop_front();
        for (auto [val, nxt] : ed[cur + 1]) {
          reverse_order[nxt] = subset_idx_prev.size();
          subset_idx_prev.push_back({val, cur == -1 ? cur : reverse_order[cur]});
          queue.push_back(nxt);
        }
      }
      if (o1 != subset_idx_prev.size()) throw;
      std::vector<uint8_t> new_subset;
      for (auto& i : adj_subset) new_subset.push_back(reverse_order[i]);
      adj_subset.swap(new_subset);
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
      Reorder();
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
    ind += VecOutput<1>(next_ids, ret + ind, [&](auto& i, uint8_t data[]) {
      if (i.first >= (1ll << 32)) throw std::out_of_range("too many boards");
      IntToBytes<uint32_t>(i.first, data);
      data[4] = i.second;
      return 5;
    });
    // non_adjs
    ind += SimpleVecOutput<1>(non_adj, ret + ind);
    // adjs
    if (use_subset) {
      ind += SimpleVecOutput<1>(adj_subset, ret + ind);
      ind += VecOutput<1>(subset_idx_prev, ret + ind, [&](auto& i, uint8_t data[]) {
        if (i.second >= 255 || i.second < -1) throw std::runtime_error("invalid prev");
        data[0] = i.first;
        data[1] = i.second == -1 ? 255 : i.second;
        //IntToBytes<uint16_t>(i.second == -1 ? 65536 : i.second, data + 1);
        return 2;
      });
    } else {
      ind += VecOutput<1>(adj, ret + ind, [&](auto& adj_item, uint8_t data[]) {
        return SimpleVecOutput<1>(adj_item, data);
      });
    }
    if (ind != sz) throw std::runtime_error("size not match");
  }

  EvaluateNodeEdges() : cell_count(), use_subset() {}
  EvaluateNodeEdges(const uint8_t data[], size_t sz) {
    size_t ind = 0;
    cell_count = data[ind++];
    use_subset = data[ind++];
    ind += VecInput<1>(next_ids, data + ind, [](auto& i, const uint8_t data[]) {
      i.first = BytesToInt<uint32_t>(data);
      i.second = data[4];
      return 5;
    });
    // non_adjs
    ind += SimpleVecInput<1>(non_adj, data + ind);
    // adjs
    if (use_subset) {
      ind += SimpleVecInput<1>(adj_subset, data + ind);
      ind += VecInput<1>(subset_idx_prev, data + ind, [](auto& i, const uint8_t data[]) {
        i.first = data[0];
        i.second = data[1];
        if (i.second == 255) i.second = -1;
        //i.second = BytesToInt<uint16_t>(data + ind + 1);
        //if (i.second == 65536) i.second = -1;
        return 2;
      });
    } else {
      ind += VecInput<1>(adj, data + ind, [](auto& adj_item, const uint8_t data[]) {
        return SimpleVecInput<1>(adj_item, data);
      });
    }
    if (ind != sz) throw std::runtime_error("size not match");
  }
};

struct PositionNodeEdges {
  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;

  std::vector<Position> nexts;
  std::vector<Position> adj;

  bool operator==(const PositionNodeEdges&) const = default;
  bool operator!=(const PositionNodeEdges&) const = default;

  size_t NumBytes() const {
    if (nexts.size() >= 256 || adj.size() >= 256) throw std::out_of_range("output too large");
    return 2 + nexts.size() * 2 + adj.size() * 2;
  }

  void GetBytes(uint8_t ret[]) const {
    size_t sz = NumBytes();
    size_t ind = 0;
    ind += VecOutput<1>(nexts, ret + ind, [&](auto& i, uint8_t data[]) {
      i.GetBytes(data);
      return 2;
    });
    ind += VecOutput<1>(adj, ret + ind, [&](auto& i, uint8_t data[]) {
      i.GetBytes(data);
      return 2;
    });
    if (ind != sz) throw std::runtime_error("size not match");
  }

  PositionNodeEdges() = default;
  PositionNodeEdges(const uint8_t data[], size_t sz) {
    size_t ind = 0;
    ind += VecInput<1>(nexts, data + ind, [](auto& i, const uint8_t data[]) {
      i = Position(data, 2);
      return 2;
    });
    ind += VecInput<1>(adj, data + ind, [](auto& i, const uint8_t data[]) {
      i = Position(data, 2);
      return 2;
    });
    if (ind != sz) throw std::runtime_error("size not match");
  }
};
