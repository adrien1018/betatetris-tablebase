#pragma GCC optimize("fast-math")

#include "move.h"

#include <cmath>
#include <set>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#pragma GCC diagnostic pop
#include "edge.h"
#include "game.h"
#include "board.h"
#include "config.h"
#include "evaluate.h"
#include "board_set.h"
#include "thread_queue.h"

#pragma GCC diagnostic ignored "-Wclass-memaccess"

namespace {

struct Stats {
  std::atomic<float> maximum;

  void Update(const float val) {
    float prev_value = maximum;
    while (prev_value < val && !maximum.compare_exchange_weak(prev_value, val));
  }
  void Clear() {
    maximum = 0.0f;
  }
} stats;

inline void NodeMoveIndexFromVec(NodeMoveIndex& ret, __m256i v) {
  alignas(32) uint32_t idx[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(idx), v);
  for (size_t i = 0; i < kPieces; i++) ret[i] = idx[i];
}

void CalculateBlock(
    const EvaluateNodeEdgesFast* edges, size_t edges_size,
    const std::vector<MoveEval>& prev,
    int level,
    MoveEval out[], NodeMoveIndex out_idx[]) {
  if (!edges_size) return;
  if (edges_size % kPieces != 0) throw std::logic_error("unexpected: not multiples of 7");
  size_t boards = edges_size / kPieces;

  MoveEval local_val[256], adj_val[256];
  __m256i adj_idx[256];
  for (size_t b = 0; b < boards; b++) {
    float ev[7] = {};
    for (size_t piece = 0; piece < kPieces; piece++) {
      auto& item = edges[b * kPieces + piece];
      if (!item.next_ids_size) continue;
      for (size_t i = 0; i < item.next_ids_size; i++) {
        auto& [next, lines] = item.next_ids[i];
        local_val[i] = prev[next];
        local_val[i] += Score(lines, level);
      }
      __m256 probs = _mm256_load_ps(kTransitionProb[piece]);
      __m256i res_idx = _mm256_setzero_si256();
      float mx_ev = 0.;
      auto Update = [&res_idx,&mx_ev,&probs](const MoveEval& cur, __m256i new_idx) {
        float cur_ev = cur.Dot(probs);
        if (cur_ev > mx_ev) mx_ev = cur_ev, res_idx = new_idx;
      };
      for (size_t i = 0; i < item.non_adj_size; i++) {
        Update(local_val[item.non_adj[i]], _mm256_set1_epi32(item.non_adj[i]));
      }
      if (item.use_subset) {
        for (size_t i = 0; i < item.subset_idx_prev_size; i++) {
          auto& [idx, prev] = item.subset_idx_prev[i];
          adj_val[i] = local_val[idx];
          if (prev != 255) adj_idx[i] = adj_val[i].MaxWith(adj_val[prev], adj_idx[i], idx);
        }
        for (size_t i = 0; i < item.adj_subset_size; i++) {
          auto idx = item.adj_subset[i];
          Update(adj_val[idx], adj_idx[idx]);
        }
      } else {
        for (size_t i = 0; i < item.adj_lst_size; i++) {
          size_t start = item.adj_lst[i], end = item.adj_lst[i+1];
          MoveEval cur = local_val[item.adj[start]];
          __m256i cur_idx = _mm256_setzero_si256();
          for (size_t j = start + 1; j < end; j++) {
            cur_idx = cur.MaxWith(local_val[item.adj[j]], cur_idx, item.adj[j]);
          }
          Update(cur, cur_idx);
        }
      }
      ev[piece] = mx_ev;
      NodeMoveIndexFromVec(out_idx[b * kPieces + piece], res_idx);
      stats.Update(mx_ev);
    }
    out[b].LoadEv(ev);
  }
}

void CalculateSameLevel(
    int group, size_t start, size_t end, const std::vector<MoveEval>& prev, int level,
    MoveEval out[], CompressedClassWriter<NodeMoveIndex>& idx_writer) {
  constexpr size_t kBatchSize = 1024;
  constexpr size_t kBlockSize = 524288;

  auto fname = EvaluateEdgePath(group, GetLevelSpeed(level));
  using Result = std::pair<size_t, size_t>;

  std::vector<NodeMoveIndex> out_idx((end - start) * kPieces);

  BS::thread_pool io_pool(kIOThreads);
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
    io_pool.push_task([&fname,&thread_queue,&works,&unfinished,&cv,&mtx,block_start,block_end,start,&prev,level,out,&out_idx]() {
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
                edges=std::move(edges),&works,&unfinished,&cv,&mtx,num_to_read,start,batch_l,batch_r,&prev,level,out,&out_idx
          ]() {
            CalculateBlock(edges.get(), num_to_read, prev, level, out + batch_l, out_idx.data() + (batch_l - start) * kPieces);
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
  idx_writer.Write(out_idx);
}

std::vector<MoveEval> CalculatePieceMoves(
    int pieces, const std::vector<MoveEval>& prev, const std::vector<size_t>& offsets) {
  int group = GetGroupByPieces(pieces);
  std::vector<MoveEval> ret(offsets.back());
  CompressedClassWriter<NodeMoveIndex> writer(MovePath(pieces), 4096 * kPieces);

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
      memset(ret.data() + start, 0x0, (offsets[i + 1] - start) * sizeof(MoveEval));
      writer.Write(NodeMoveIndex{}, (offsets[i + 1] - start) * kPieces);
      start = offsets[i + 1];
      continue;
    }
    int level = GetLevelByLines(lines);
    if (cur_level != 0) {
      spdlog::debug("Calculate group {} lvl {}: {} - {}", group, cur_level, start, offsets[i]);
      CalculateSameLevel(group, start, offsets[i], prev, cur_level, ret.data(), writer);
      start = offsets[i];
    }
    cur_level = level;
  }
  if (cur_level != 0) {
    spdlog::debug("Calculate group {} lvl {}: {} - {}", group, cur_level, start, last);
    CalculateSameLevel(group, start, last, prev, cur_level, ret.data(), writer);
  }
  if (last < offsets.back()) {
    memset(ret.data() + last, 0x0, (offsets.back() - last) * sizeof(MoveEval));
    writer.Write(NodeMoveIndex{}, (offsets.back() - last) * kPieces);
  }
  std::vector<float> ev(7);
  ret[0].GetEv(ev.data());
  spdlog::debug("Finish piece {}: max_val {}, val0 {}", pieces, stats.maximum.load(), ev);
  return ret;
}

void MergeRanges(int group, int pieces_l, int pieces_r, const std::vector<size_t>& offset, bool delete_after) {
  int orig_pieces_l = pieces_l;
  spdlog::info("Merging group {}", group);
  while (GetGroupByPieces(pieces_l) != group) pieces_l++;
  std::vector<CompressedClassReader<NodeMoveIndex>> readers;
  for (int pieces = pieces_l; pieces < pieces_r; pieces += kGroups) {
    readers.emplace_back(MovePath(pieces));
  }
  if (readers.empty()) return;
  CompressedClassWriter<NodeMoveIndexRange> writer(MoveRangePath(orig_pieces_l, pieces_r, group), 4096 * kPieces, -2);
  std::vector<NodeMoveIndex> buf(readers.size());
  for (size_t i = 0; i < offset.size() - 1; i++) {
    int start_cells = pieces_l * 4 - int(i * 10 + group * 2);
    if (start_cells % 10 != 0) throw std::logic_error("unexpected");
    int start_lines = start_cells / 10;
    uint8_t start_lines_idx = (uint32_t)start_lines / 2;
    size_t begin = 0;
    size_t end = std::min(buf.size(), (size_t)(kLineCap - start_lines + 1) / 2);
    if (start_lines < 0) {
      start_lines_idx = 0;
      begin = (-start_lines + 1) / 2;
    }
    for (size_t x = 0; x < (offset[i + 1] - offset[i]) * kPieces; x++) {
      for (size_t j = 0; j < readers.size(); j++) readers[j].ReadOne(&buf[j]);
      writer.Write(NodeMoveIndexRange(buf.begin() + begin, buf.begin() + end, start_lines_idx));
    }
  }
  spdlog::info("Group {} merged", group);
  if (delete_after) {
    readers.clear();
    for (int pieces = pieces_l; pieces < pieces_r; pieces += kGroups) {
      std::filesystem::remove(MovePath(pieces));
      std::filesystem::remove(std::string(MovePath(pieces)) + ".index");
    }
  }
}

} // namespace

void RunCalculateMoves(int start_pieces, int end_pieces) {
  std::vector<size_t> offsets[kGroups];
  for (int i = 0; i < kGroups; i++) offsets[i] = GetBoardCountOffset(i);

  std::vector<MoveEval> values;
  if (start_pieces == -1) {
    size_t max_cells = 0;
    for (int i = 0; i < kGroups; i++) {
      max_cells = std::max(max_cells, (offsets[i].size() - 1) * 10 + i * 2);
    }
    start_pieces = (kLineCap * 10 + max_cells + 3) / 4;
    int start_group = GetGroupByPieces(start_pieces);
    values.resize(offsets[start_group].back());
    memset(values.data(), 0x0, values.size() * sizeof(MoveEval));
  } else {
    int start_group = GetGroupByPieces(start_pieces);
    values = ReadValuesEvOnly(start_pieces, offsets[start_group].back());
    if (values.size() != offsets[start_group].back()) throw std::length_error("initial value file incorrect");
  }

  for (int pieces = start_pieces - 1; pieces >= end_pieces; pieces--) {
    values = CalculatePieceMoves(pieces, values, offsets[GetGroupByPieces(pieces)]);
  }
}

void MergeRanges(int pieces_l, int pieces_r, bool delete_after) {
  int threads = std::min(kParallel, kGroups);
  BS::thread_pool pool(threads);
  pool.parallelize_loop(0, kGroups, [&](int l, int r){
    for (int group = l; group < r; group++) {
      MergeRanges(group, pieces_l, pieces_r, GetBoardCountOffset(group), delete_after);
    }
  }).wait();
}
