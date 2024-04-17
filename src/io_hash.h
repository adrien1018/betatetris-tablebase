#pragma once

#include <fstream>
#include <optional>
#include "io.h"

template <class Key, class Val, class Hash = std::hash<Key>>
void WriteHashMap(const std::string& fname, std::vector<std::pair<Key, Val>>&& vals, size_t num_buckets) {
  constexpr size_t kBufferSize = 1048576;
  constexpr size_t kIndexBufferSize = 131072;

  Hash hash;
  auto nhash = [&hash,num_buckets](const Key& k) { return hash(k) % num_buckets; };
  std::sort(vals.begin(), vals.end(), [&nhash](const auto& a, const auto& b) {
    return nhash(a.first) < nhash(b.first);
  });

  uint64_t current_offset = 0;
  std::vector<uint8_t> buf, buf_ind;
  std::ofstream fout(fname), fout_ind(fname + ".index");
  auto Flush = [&current_offset,&buf,&fout]() {
    if (!fout.write(reinterpret_cast<const char*>(buf.data()), buf.size())) {
      throw std::runtime_error("write failed");
    }
    current_offset += buf.size();
    buf.clear();
  };
  auto FlushIndex = [&buf_ind,&fout_ind]() {
    if (!fout_ind.write(reinterpret_cast<const char*>(buf_ind.data()), buf_ind.size())) {
      throw std::runtime_error("write failed");
    }
    buf_ind.clear();
  };
  auto WriteIndex = [&buf_ind,&buf,&current_offset]() {
    size_t old_sz = buf_ind.size();
    buf_ind.resize(old_sz + sizeof(uint64_t));
    IntToBytes<uint64_t>(current_offset + buf.size(), buf_ind.data() + old_sz);
  };
  buf_ind.resize(sizeof(uint64_t));
  IntToBytes<uint64_t>(num_buckets, buf_ind.data());
  for (size_t bucket = 0, i = 0; bucket < num_buckets; bucket++) {
    WriteIndex();
    if (buf_ind.size() >= kIndexBufferSize) FlushIndex();
    for (; i < vals.size() && nhash(vals[i].first) == bucket; i++) {
      io_internal::WriteToBuf(buf, vals[i].first);
      io_internal::WriteToBuf(buf, vals[i].second);
      if (buf.size() >= kBufferSize) Flush();
    }
  }
  Flush();
  WriteIndex();
  FlushIndex();
}

template <class Key, class Val, class Hash = std::hash<Key>>
class HashMapReader {
  std::ifstream fin, fin_index;
  size_t num_buckets;
  Hash hash;
 public:
  HashMapReader(const std::string& fname) {
    fin.rdbuf()->pubsetbuf(nullptr, 0);
    fin.open(fname);
    if (!fin.is_open()) throw std::runtime_error("cannot open file");
    fin_index.rdbuf()->pubsetbuf(nullptr, 0);
    fin_index.open(fname + ".index");
    if (!fin_index.is_open()) throw std::runtime_error("cannot open index file");
    uint8_t sz_buf[8];
    if (!(fin_index.read(reinterpret_cast<char*>(sz_buf), 8) && fin_index.gcount() == 8)) {
      throw std::runtime_error("invalid index file");
    }
    num_buckets = BytesToInt<uint64_t>(sz_buf);
  }

  std::optional<Val> operator[](const Key& k) {
    size_t bucket = hash(k) % num_buckets;
    uint8_t sz_buf[16];
    fin_index.clear();
    fin_index.seekg((bucket + 1) * sizeof(uint64_t));
    if (!(fin_index.read(reinterpret_cast<char*>(sz_buf), 16) && fin_index.gcount() == 16)) {
      throw std::runtime_error("invalid index file");
    }
    uint64_t start = BytesToInt<uint64_t>(sz_buf), end = BytesToInt<uint64_t>(sz_buf + 8);
    if (start == end) return std::nullopt;
    std::vector<uint8_t> buf(end - start);
    fin.clear();
    fin.seekg(start);
    if (!(fin.read(reinterpret_cast<char*>(buf.data()), buf.size()) &&
          (size_t)fin.gcount() == buf.size())) {
      throw std::runtime_error("invalid file");
    }
    size_t offset = 0;
    while (offset < buf.size()) {
      auto [key_sz, key_offset] = io_internal::GetNextSize<Key>(buf.data() + offset);
      offset += key_offset;
      auto [val_sz, val_offset] = io_internal::GetNextSize<Val>(buf.data() + offset + key_sz);
      if (k == Key(buf.data() + offset, key_sz)) {
        return Val(const_cast<const uint8_t*>(buf.data() + offset + key_sz + val_offset), val_sz);
      } else {
        offset += key_sz + val_offset + val_sz;
      }
    }
    return std::nullopt;
  }
};
