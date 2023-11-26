#pragma once

#include <cstdint>
#include <deque>
#include <bitset>
#include <vector>
#include <stdexcept>
#include <tsl/hopscotch_map.h>

#include "hash.h"
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

  // ret[new_idx] = {old_idx...}
  std::vector<std::vector<uint8_t>> ReduceAdj() {
    using Key = std::bitset<256>;
    tsl::hopscotch_map<Key, std::vector<uint8_t>, std::hash<Key>, std::equal_to<Key>,
                       std::allocator<std::pair<Key, std::vector<uint8_t>>>, 30, true> mp;
    for (size_t i = 0; i < adj.size(); i++) {
      std::bitset<256> bs{};
      for (const auto& j : adj[i]) bs[j] = true;
      mp[bs].push_back(i);
    }
    std::vector<std::vector<uint8_t>> ret, new_adj;
    for (auto& i : mp) {
      new_adj.push_back(std::move(adj[i.second[0]]));
      ret.push_back(std::move(i.second));
    }
    adj.swap(new_adj);
    return ret;
  }

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

template <int kBufSize>
struct EvaluateNodeEdgesFastTmpl {
  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;

  uint8_t* base_ptr;
  // edge for`evaluating; no position information available
  uint8_t cell_count;
  bool use_subset;
  // (next board ID, lines)
  std::pair<uint64_t, uint8_t>* next_ids;
  size_t next_ids_size;
  // indices of next_ids
  uint8_t* non_adj;
  size_t non_adj_size;
  uint32_t* adj_lst; // length=adj_lst_size+1
  size_t adj_lst_size;
  uint8_t* adj;
  // subset construction for faster calculation
  uint8_t* adj_subset;
  size_t adj_subset_size;
  std::pair<uint8_t, int>* subset_idx_prev; // (idx, prev)
  size_t subset_idx_prev_size;

  uint8_t buf[kBufSize];
  size_t buf_size;

  bool operator==(const EvaluateNodeEdgesFastTmpl<kBufSize>& x) const {
    return std::equal(buf, buf + buf_size, x.buf);
  }
  bool operator!=(const EvaluateNodeEdgesFastTmpl<kBufSize>& x) const { return !(*this == x); }

  bool operator==(const EvaluateNodeEdges& x) const {
    if (!(cell_count == x.cell_count && use_subset == x.use_subset &&
          next_ids_size == x.next_ids.size() &&
          non_adj_size == x.non_adj.size() &&
          std::equal(next_ids, next_ids + next_ids_size, x.next_ids.begin()) &&
          std::equal(non_adj, non_adj + non_adj_size, x.non_adj.begin()))) {
      return false;
    }
    if (use_subset) {
      return adj_subset_size == x.adj_subset.size() &&
        subset_idx_prev_size == x.subset_idx_prev.size() &&
        std::equal(adj_subset, adj_subset + adj_subset_size, x.adj_subset.begin()) &&
        std::equal(subset_idx_prev, subset_idx_prev + subset_idx_prev_size, x.subset_idx_prev.begin());
    }
    if (adj_lst_size != x.adj.size()) return false;
    for (size_t i = 0; i < adj_lst_size; i++) {
      if (!(adj_lst[i+1] - adj_lst[i] == x.adj[i].size() &&
            std::equal(adj + adj_lst[i], adj + adj_lst[i+1], x.adj[i].begin()))) return false;
    }
    return true;
  }

  size_t NumBytes() const {
    throw std::runtime_error("should not use in write");
  }

  void GetBytes(uint8_t ret[]) const {
    throw std::runtime_error("should not use in write");
  }

  EvaluateNodeEdgesFastTmpl() : cell_count(), use_subset() {}
  EvaluateNodeEdgesFastTmpl(const uint8_t data[], size_t sz) {
    base_ptr = buf;
    size_t ind = 0, buf_ind = 0;
    cell_count = data[ind++];
    use_subset = data[ind++];
    // nexts
    next_ids_size = data[ind++];
    next_ids = reinterpret_cast<decltype(next_ids)>(buf + buf_ind);
    for (size_t i = 0; i < next_ids_size; i++) {
      next_ids[i].first = BytesToInt<uint32_t>(data + ind);
      next_ids[i].second = data[ind + 4];
      ind += 5;
    }
    buf_ind += sizeof(decltype(*next_ids)) * next_ids_size;
    // non_adjs
    non_adj_size = data[ind++];
    non_adj = buf + buf_ind;
    memcpy(non_adj, data + ind, non_adj_size);
    ind += non_adj_size;
    buf_ind += non_adj_size;
    // adjs
    if (use_subset) {
      adj_subset_size = data[ind++];
      adj_subset = buf + buf_ind;
      memcpy(adj_subset, data + ind, adj_subset_size);
      ind += adj_subset_size;
      buf_ind += adj_subset_size;

      buf_ind = (buf_ind + 7) / 8 * 8; // align
      subset_idx_prev_size = data[ind++];
      subset_idx_prev = reinterpret_cast<decltype(subset_idx_prev)>(buf + buf_ind);
      for (size_t i = 0; i < subset_idx_prev_size; i++) {
        auto& item = subset_idx_prev[i];
        item.first = data[ind];
        item.second = data[ind + 1];
        if (item.second == 255) item.second = -1;
        ind += 2;
      }
      buf_ind += sizeof(decltype(*subset_idx_prev)) * subset_idx_prev_size;
    } else {
      buf_ind = (buf_ind + 7) / 8 * 8; // align
      adj_lst_size = data[ind++];
      adj_lst = reinterpret_cast<decltype(adj_lst)>(buf + buf_ind);
      buf_ind += sizeof(decltype(*adj_lst)) * (adj_lst_size + 1);
      adj = buf + buf_ind;
      uint32_t offset = 0;
      for (size_t i = 0; i < adj_lst_size; i++) {
        uint8_t sz = data[ind++];
        adj_lst[i] = offset;
        memcpy(adj + offset, data + ind, sz);
        offset += sz;
        ind += sz;
      }
      adj_lst[adj_lst_size] = offset;
      buf_ind += offset;
    }
    buf_size = buf_ind;
    if (ind != sz) throw std::runtime_error("size not match");
    if (buf_size >= sizeof(buf)) throw std::runtime_error("buffer overflow");
  }
};

using EvaluateNodeEdgesFast = EvaluateNodeEdgesFastTmpl<2560>;

struct PositionNodeEdges {
  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 2;

  std::vector<Position> nexts;
  std::vector<std::vector<Position>> adj;

  bool operator==(const PositionNodeEdges&) const = default;
  bool operator!=(const PositionNodeEdges&) const = default;

  size_t NumBytes() const {
    if (nexts.size() >= 256 || adj.size() >= 256) throw std::out_of_range("output too large");
    size_t ret = 2 + nexts.size() * 2;
    for (auto& i : adj) ret += 1 + i.size() * 2;
    return ret;
  }

  void GetBytes(uint8_t ret[]) const {
    size_t sz = NumBytes();
    size_t ind = 0;
    ind += VecOutput<1>(nexts, ret + ind, [&](auto& i, uint8_t data[]) {
      i.GetBytes(data);
      return 2;
    });
    ind += VecOutput<1>(adj, ret + ind, [&](auto& i, uint8_t data[]) {
      return VecOutput<1>(i, data, [&](auto& j, uint8_t data2[]) {
        j.GetBytes(data2);
        return 2;
      });
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
      return VecInput<1>(i, data, [](auto& j, const uint8_t data[]) {
        j = Position(data, 2);
        return 2;
      });
    });
    if (ind != sz) throw std::runtime_error("size not match");
  }
};
