#pragma GCC optimize("fast-math")

#include "evaluate.h"

#include <cmath>
#include <set>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop
#include "edge.h"
#include "board.h"
#include "config.h"
#include "board_set.h"
#include "move_search.h"
#include "thread_queue.h"

#pragma GCC diagnostic ignored "-Wclass-memaccess"

namespace {

constexpr int kLineCap = LINE_CAP;

alignas(32) constexpr float kTransitionProb[][8] = {
  {1./32, 5./32, 6./32, 5./32, 5./32, 5./32, 5./32}, // T
  {6./32, 1./32, 5./32, 5./32, 5./32, 5./32, 5./32}, // J
  {5./32, 6./32, 1./32, 5./32, 5./32, 5./32, 5./32}, // Z
  {5./32, 5./32, 5./32, 2./32, 5./32, 5./32, 5./32}, // O
  {5./32, 5./32, 5./32, 5./32, 2./32, 5./32, 5./32}, // S
  {6./32, 5./32, 5./32, 5./32, 5./32, 1./32, 5./32}, // L
  {5./32, 5./32, 5./32, 5./32, 6./32, 5./32, 1./32}, // I
};

constexpr int GetLevelByLines(int lines) {
  if (lines < 130) return 18;
  return lines / 10 + 6;
}
static_assert(GetLevelByLines(130) == 19);
static_assert(GetLevelByLines(230) == 29);
static_assert(GetLevelByLines(330) == 39);

constexpr Level GetLevelSpeed(int level) {
  if (level == 18) return kLevel18;
  if (level < 29) return kLevel19;
  if (level < 39) return kLevel29;
  return kLevel39;
}

constexpr int GetGroupByPieces(int pieces) {
  return pieces * 4 / 2 % 5;
}
static_assert(GetGroupByPieces(0) == 0);
static_assert(GetGroupByPieces(1) == 2);
static_assert(GetGroupByPieces(4) == 3);
static_assert(GetGroupByPieces(9) == 3);

constexpr int Score(int lines, int level) {
  constexpr int kTable[] = {0, 40, 100, 300, 1200};
  return kTable[lines] * (level + 1);
}


struct Stats {
  static constexpr int kBucketSize = 32;
  static constexpr int kMaximum = 2400000;
  std::array<std::atomic_long, kMaximum / kBucketSize> distribution;
  std::atomic<float> maximum;

  void Update(const float val) {
    int bucket = (int)val / kBucketSize;
    distribution[bucket]++;
    float prev_value = maximum;
    while (prev_value < val && !maximum.compare_exchange_weak(prev_value, val));
  }
  void Clear() {
    memset(distribution.data(), 0x0, sizeof(distribution));
    maximum = 0.0f;
  }
} stats;

void CalculateBlock(
    const EvaluateNodeEdgesFast* edges, size_t edges_size,
    const std::vector<NodeEval>& prev,
    int level,
    NodeEval out[]) {
  if (!edges_size) return;
  if (edges_size % kPieces != 0) throw std::logic_error("unexpected: not multiples of 7");
  size_t boards = edges_size / kPieces;

  NodeEval local_val[256], adj_val[256];
  for (size_t b = 0; b < boards; b++) {
    float ev[7] = {}, var[7] = {};
    for (size_t piece = 0; piece < kPieces; piece++) {
      auto& item = edges[b * kPieces + piece];
      if (!item.next_ids_size) continue;
      for (size_t i = 0; i < item.next_ids_size; i++) {
        auto& [next, lines] = item.next_ids[i];
        local_val[i] = prev[next];
        local_val[i] += Score(lines, level);
      }
      __m256 probs = _mm256_load_ps(kTransitionProb[piece]);
      NodeEval result(_mm256_setzero_ps(), _mm256_setzero_ps());
      float mx_ev = 0.;
      auto Update = [&result,&mx_ev,&probs](const NodeEval& cur) {
        float cur_ev = cur.Dot(probs);
        if (cur_ev > mx_ev) mx_ev = cur_ev, result = cur;
      };
      for (size_t i = 0; i < item.non_adj_size; i++) {
        Update(local_val[item.non_adj[i]]);
      }
      if (item.use_subset) {
        for (size_t i = 0; i < item.subset_idx_prev_size; i++) {
          auto& [idx, prev] = item.subset_idx_prev[i];
          adj_val[i] = local_val[idx];
          if (prev != 255) adj_val[i].MaxWith(adj_val[prev]);
        }
        for (size_t i = 0; i < item.adj_subset_size; i++) {
          Update(adj_val[item.adj_subset[i]]);
        }
      } else {
        for (size_t i = 0; i < item.adj_lst_size; i++) {
          size_t start = item.adj_lst[i], end = item.adj_lst[i+1];
          NodeEval cur = local_val[item.adj[start]];
          for (size_t j = start + 1; j < end; j++) cur.MaxWith(local_val[item.adj[j]]);
          Update(cur);
        }
      }
      ev[piece] = mx_ev;
      var[piece] = result.DotVar(probs, mx_ev);
      stats.Update(mx_ev);
    }
    out[b].LoadEv(ev);
    out[b].LoadVar(var);
  }
}

// https://stackoverflow.com/questions/20843271/passing-a-non-copyable-closure-object-to-stdfunction-parameter
template< class F >
auto make_copyable_function(F&& f) {
  using dF=std::decay_t<F>;
  auto spf = std::make_shared<dF>( std::forward<F>(f) );
  return [spf](auto&&... args)->decltype(auto) {
    return (*spf)( decltype(args)(args)... );
  };
}

void CalculateSameLevel(
    int group, size_t start, size_t end, int io_threads,
    const std::vector<NodeEval>& prev, int level,
    NodeEval out[]) {
  constexpr size_t kBatchSize = 1024;
  constexpr size_t kBlockSize = 524288;

  auto fname = EvaluateEdgePath(group, GetLevelSpeed(level));
  using Result = std::pair<size_t, size_t>;

  BS::thread_pool io_pool(io_threads);
  auto thread_queue = MakeThreadQueue<Result>(kParallel,
      [&](Result range) {
        /* collect stats */
      });

  std::mutex mtx;
  std::condition_variable cv;
  size_t unfinished = 0;
  std::deque<std::function<Result()>> works;

  std::vector<std::thread> thrs;
  for (size_t block_start = start; block_start < end; block_start += kBlockSize) {
    size_t block_end = std::min(end, block_start + kBlockSize);
    unfinished++;
    io_pool.push_task([&fname,&thread_queue,&works,&unfinished,&cv,&mtx,block_start,block_end,&prev,level,out]() {
      CompressedClassReader<EvaluateNodeEdgesFast> reader(fname);
      reader.Seek(block_start * kPieces);
      for (size_t batch_l = block_start; batch_l < block_end; batch_l += kBatchSize) {
        size_t batch_r = std::min(block_end, batch_l + kBatchSize);
        size_t num_to_read = (batch_r - batch_l) * kPieces;
        std::unique_ptr<EvaluateNodeEdgesFast[]> edges(new EvaluateNodeEdgesFast[num_to_read]);
        if (num_to_read != reader.ReadBatch(edges.get(), num_to_read)) throw std::runtime_error("read failure");
        {
          std::lock_guard lck(mtx);
          works.push_back(make_copyable_function([
                edges=std::move(edges),&works,&unfinished,&cv,&mtx,num_to_read,batch_l,batch_r,&prev,level,out
          ]() {
            CalculateBlock(edges.get(), num_to_read, prev, level, out + batch_l);
            return std::make_pair(batch_l, batch_r);
          }));
        }
        cv.notify_one();
      }
      {
        std::lock_guard lck(mtx);
        unfinished--;
      }
      cv.notify_one();
    });
  }
  {
    std::unique_lock lck(mtx);
    while (unfinished) {
      cv.wait(lck, [&]{ return unfinished == 0 || works.size(); });
      while (works.size()) {
        thread_queue.Push(std::move(works.front()));
        works.pop_front();
      }
    }
  }
  io_pool.wait_for_tasks();
  thread_queue.WaitAll();
}

std::vector<NodeEval> CalculatePiece(
    int pieces, int io_threads, const std::vector<NodeEval>& prev, const std::vector<size_t>& offsets) {
  int group = GetGroupByPieces(pieces);
  std::vector<NodeEval> ret(offsets.back());

  spdlog::info("Start calculate piece {}", pieces);
  stats.Clear();
  size_t start = 0, last = offsets.back();
  int cur_level = 0; // 0 -> uninitialized
  for (size_t i = 0; i < offsets.size() - 1; i++) {
    int cells = pieces * 4 - int(i * 10 + group * 2);
    if (cells < 0) {
      last = offsets[i];
      break;
    }
    if (cells % 10) throw std::logic_error("unexpected: cells incorrect");
    int lines = cells / 10;
    if (lines >= kLineCap) {
      // lines will decrease as i increase, so this only happen at the start of the loop
      memset(ret.data() + start, 0x0, (offsets[i + 1] - start) * sizeof(NodeEval));
      start = offsets[i + 1];
      continue;
    }
    int level = GetLevelByLines(lines);
    if (level != cur_level && cur_level != 0) {
      spdlog::debug("Calculate group {} lvl {}: {} - {}", group, cur_level, start, offsets[i]);
      CalculateSameLevel(group, start, offsets[i], io_threads, prev, cur_level, ret.data());
      start = offsets[i];
    }
    cur_level = level;
  }
  if (cur_level != 0) {
    spdlog::debug("Calculate group {} lvl {}: {} - {}", group, cur_level, start, last);
    CalculateSameLevel(group, start, last, io_threads, prev, cur_level, ret.data());
  }
  {
    MkdirForFile(ValueStatsPath(pieces));
    std::ofstream fout(ValueStatsPath(pieces));
    for (size_t i = 0; i <= (size_t)stats.maximum.load() / Stats::kBucketSize; i++) {
      if (long x = stats.distribution[i].load()) {
        fout << i * Stats::kBucketSize << ' ' << x << '\n';
      }
    }
  }
  std::vector<float> ev(7), var(7);
  ret[0].GetEv(ev.data());
  ret[0].GetVar(var.data());
  for (auto& i : var) i = std::sqrt(i);
  spdlog::debug("Finish piece {}: max_val {}, val0 {} {}", pieces, stats.maximum.load(), ev, var);
  return ret;
}

} // namespace

void RunEvaluate(int io_threads, int start_pieces, const std::vector<int>& output_locations) {
  std::vector<size_t> offsets[5];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  std::vector<NodeEval> values;
  if (start_pieces == -1) {
    size_t max_cells = 0;
    for (int i = 0; i < kGroups; i++) {
      max_cells = std::max(max_cells, (offsets[i].size() - 1) * 10 + i * 2);
    }
    start_pieces = (kLineCap * 10 + max_cells + 3) / 4;
    int start_group = GetGroupByPieces(start_pieces);
    values.resize(offsets[start_group].back());
    memset(values.data(), 0x0, values.size() * sizeof(NodeEval));
  } else {
    int start_group = GetGroupByPieces(start_pieces);
    CompressedClassReader<NodeEval> reader(ValuePath(start_pieces));
    values = reader.ReadBatch(offsets[start_group].back());
    if (values.size() != offsets[start_group].back()) throw std::length_error("initial value file incorrect");
  }

  // really lazy here
  std::set<int> location_set(output_locations.begin(), output_locations.end());
  int last_output = *location_set.begin();
  for (int pieces = start_pieces - 1; pieces >= last_output; pieces--) {
    values = CalculatePiece(pieces, io_threads, values, offsets[GetGroupByPieces(pieces)]);
    if (location_set.count(pieces)) {
      spdlog::info("Writing values of piece {}", pieces);
      CompressedClassWriter<NodeEval> writer(ValuePath(pieces), 2048);
      writer.Write(values);
    }
  }
}
