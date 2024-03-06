#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>
#include <optional>

#include <zstd.h>
#include "thread_pool.hpp"

struct CompressorBase {
  virtual ~CompressorBase() {
    if (RemainingBlocks()) puts("Warning: Compressor destructed with blocks unread");
  }
  virtual void CompressBlock(std::vector<uint8_t>&& block) {}
  virtual size_t RemainingBlocks() const { return 0; }
  virtual std::vector<uint8_t> GetResultBlock() { return {}; }
  virtual std::optional<std::vector<uint8_t>> GetResultBlockNoWait() { return std::nullopt; }
};

class DefaultZstdCompressor : public CompressorBase {
  std::unique_ptr<ZSTD_CCtx, size_t(*)(ZSTD_CCtx*)> zstd_ctx;
  std::vector<uint8_t> result;
  int compress_level;
  bool has_result;
 public:
  DefaultZstdCompressor(int compress_level = -4) :
      zstd_ctx(ZSTD_createCCtx(), ZSTD_freeCCtx), compress_level(compress_level), has_result(false) {
    if (!zstd_ctx) throw std::runtime_error("zstd initialize failed");
  }
  ~DefaultZstdCompressor() {}

  void CompressBlock(std::vector<uint8_t>&& block) {
    if (has_result) throw std::runtime_error("previous block not obtained");
    size_t clear_size = ZSTD_compressBound(block.size());
    result.resize(clear_size);
    size_t nlen = ZSTD_compressCCtx(
        zstd_ctx.get(), result.data(), clear_size, block.data(), block.size(), compress_level);
    if (ZSTD_isError(nlen)) throw std::runtime_error("zstd compress failed");
    result.resize(nlen);
    has_result = true;
  }

  size_t RemainingBlocks() const override { return has_result; }

  std::vector<uint8_t> GetResultBlock() override {
    has_result = false;
    return std::move(result);
  }

  std::optional<std::vector<uint8_t>> GetResultBlockNoWait() override {
    if (!has_result) return std::nullopt;
    return GetResultBlock();
  }
};

class ParallelZstdCompressor : public CompressorBase {
  std::deque<std::future<std::vector<uint8_t>>> results;
  BS::thread_pool pool;
  int compress_level;
 public:
  ParallelZstdCompressor(int parallel, int compress_level = -4) : pool(parallel), compress_level(compress_level) {}
  ~ParallelZstdCompressor() {}

  void CompressBlock(std::vector<uint8_t>&& block) {
    results.push_back(pool.submit([compress_level=compress_level,block=std::move(block)]() {
      std::unique_ptr<ZSTD_CCtx, size_t(*)(ZSTD_CCtx*)> zstd_ctx(ZSTD_createCCtx(), ZSTD_freeCCtx);
      size_t clear_size = ZSTD_compressBound(block.size());
      std::vector<uint8_t> result(clear_size);
      size_t nlen = ZSTD_compressCCtx(
          zstd_ctx.get(), result.data(), clear_size, block.data(), block.size(), compress_level);
      if (ZSTD_isError(nlen)) throw std::runtime_error("zstd compress failed");
      result.resize(nlen);
      return result;
    }));
  }

  size_t RemainingBlocks() const override { return results.size(); }

  std::vector<uint8_t> GetResultBlock() override {
    auto result = results.front().get();
    results.pop_front();
    return result;
  }

  std::optional<std::vector<uint8_t>> GetResultBlockNoWait() override {
    if (!results.empty() || !IsReady(results.front())) return std::nullopt;
    return GetResultBlock();
  }
};
