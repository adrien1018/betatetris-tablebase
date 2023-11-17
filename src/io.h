#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <stdexcept>

#include "files.h"
#include "constexpr_helpers.h"

namespace io_internal {

template <class T>
struct ClassIOAttr {
  static constexpr bool kIsConstSize = T::kIsConstSize;
  static constexpr size_t kSizeNumberBytes = [](){
    if constexpr (kIsConstSize) {
      return 0;
    } else {
      return T::kSizeNumberBytes;
    }
  }();
  static constexpr size_t kBufferSize = [](){
    if constexpr (kIsConstSize) {
      return (1048575 / T::NumBytes() + 1) * T::NumBytes();
    } else {
      return 1048576;
    }
  }();
};

template <class T>
class ClassWriterImpl {
 protected:
  static constexpr size_t kIsConstSize = ClassIOAttr<T>::kIsConstSize;
  static constexpr size_t kSizeNumberBytes = ClassIOAttr<T>::kSizeNumberBytes;
  static constexpr size_t kBufferSize = ClassIOAttr<T>::kBufferSize;
  static constexpr size_t kIndexBufferSize = 131072;

  std::vector<uint8_t> buf;
  std::vector<uint64_t> inds;
  size_t current;
  size_t items_per_index;
  size_t current_written_size;
  std::ofstream fout;
  std::ofstream fout_ind;
  bool moved;

  void Flush() {
    if (!fout.write(reinterpret_cast<const char*>(buf.data()), buf.size())) {
      throw std::runtime_error("write failed");
    }
    current_written_size = fout.tellp();
    buf.clear();
  }

  void FlushIndex() {
    if (!HasIndex()) return;
    std::vector<uint8_t> out(inds.size() * sizeof(uint64_t));
    for (size_t i = 0; i < inds.size(); i++) {
      IntToBytes<uint64_t>(inds[i], out.data() + (i * sizeof(uint64_t)));
    }
    inds.clear();
    if (!fout_ind.write(reinterpret_cast<const char*>(out.data()), out.size())) {
      throw std::runtime_error("write failed");
    }
  }
 public:
  ClassWriterImpl(const std::string& fname, size_t items_per_index) :
      current(0), items_per_index(items_per_index), current_written_size(0), moved(false) {
    static_assert(kIsConstSize || (kSizeNumberBytes >= 1 && kSizeNumberBytes <= 8));
    MkdirForFile(fname);
    fout.open(fname, std::ios_base::out | std::ios_base::trunc);
    if (!fout.is_open()) throw std::runtime_error("cannot open file");
    if (HasIndex()) {
      fout_ind.open(fname + ".index", std::ios_base::out | std::ios_base::trunc);
      if (!fout_ind.is_open()) throw std::runtime_error("cannot open index file");
      inds.push_back(items_per_index);
    }
    buf.reserve(kBufferSize);
  }
  ClassWriterImpl(const ClassWriterImpl&) = delete;
  ClassWriterImpl(ClassWriterImpl&& x) :
      buf(std::move(x.buf)), inds(std::move(x.inds)),
      current(x.current), items_per_index(x.items_per_index),
      current_written_size(x.current_written_size),
      fout(std::move(x.fout)), fout_ind(std::move(x.fout_ind)), moved(false) {
    x.moved = true;
  }

  ~ClassWriterImpl() {
    if (moved) return;
    inds.push_back(ByteSize());
    Flush();
    FlushIndex();
  }

  bool HasIndex() const {
    return !kIsConstSize && items_per_index >= 1;
  }

  size_t Size() const {
    return current;
  }

  size_t ByteSize() {
    return current_written_size + buf.size();
  }
};

template <class T>
class ClassReaderImpl {
 protected:
  static constexpr size_t kIsConstSize = ClassIOAttr<T>::kIsConstSize;
  static constexpr size_t kSizeNumberBytes = ClassIOAttr<T>::kSizeNumberBytes;
  static constexpr size_t kBufferSize = ClassIOAttr<T>::kBufferSize;

  std::vector<uint8_t> buf;
  size_t current;
  size_t items_per_index;
  bool eof;
  std::ifstream fin;
  std::ifstream fin_index;

  void ReadUntilSize(size_t sz, size_t buf_size) {
    size_t old_sz = buf.size();
    sz = std::max(sz, buf_size);
    if (sz <= old_sz) return;
    buf.resize(sz);
    fin.read(reinterpret_cast<char*>(buf.data() + old_sz), sz - old_sz);
    buf.resize(old_sz + fin.gcount());
  }

  void ParseBufSize(size_t& buf_size) {
    if (buf_size == std::string::npos) buf_size = kBufferSize;
  }
 public:
  ClassReaderImpl(const std::string& fname, bool check_index) :
      current(0), items_per_index(0), eof(false) {
    static_assert(kIsConstSize || (kSizeNumberBytes >= 1 && kSizeNumberBytes <= 8));
    fin.rdbuf()->pubsetbuf(nullptr, 0);
    fin.open(fname);
    if (!fin.is_open()) throw std::runtime_error("cannot open file");
    uint8_t sz_buf[8] = {};
    if (check_index) {
      fin_index.rdbuf()->pubsetbuf(nullptr, 0);
      fin_index.open(fname + ".index");
      if (fin_index.is_open()) {
        if (fin_index.read(reinterpret_cast<char*>(sz_buf), 8) && fin_index.gcount() == 8) {
          items_per_index = BytesToInt<uint64_t>(sz_buf);
        }
        if (!items_per_index) throw std::runtime_error("invalid index file");
      }
    }
    buf.reserve(kBufferSize);
  }
  ClassReaderImpl(const ClassReaderImpl&) = delete;
  ClassReaderImpl(ClassReaderImpl&&) = default;

  bool HasIndex() const {
    return !kIsConstSize && items_per_index >= 1;
  }

  size_t Position() const {
    return current;
  }
};

} // namespace io_internal

struct ReadError : std::out_of_range {
  using std::out_of_range::out_of_range;
};

template <class T>
class ClassWriter : public io_internal::ClassWriterImpl<T> {
  using io_internal::ClassWriterImpl<T>::kIsConstSize;
  using io_internal::ClassWriterImpl<T>::kSizeNumberBytes;
  using io_internal::ClassWriterImpl<T>::kBufferSize;
  using io_internal::ClassWriterImpl<T>::kIndexBufferSize;
  using io_internal::ClassWriterImpl<T>::Flush;
  using io_internal::ClassWriterImpl<T>::FlushIndex;
  using io_internal::ClassWriterImpl<T>::current;
  using io_internal::ClassWriterImpl<T>::inds;
  using io_internal::ClassWriterImpl<T>::buf;
  using io_internal::ClassWriterImpl<T>::items_per_index;
 public:
  using io_internal::ClassWriterImpl<T>::HasIndex;
  using io_internal::ClassWriterImpl<T>::ByteSize;

  ClassWriter(const std::string& fname, size_t items_per_index = 1024) :
      io_internal::ClassWriterImpl<T>(fname, kIsConstSize ? 0 : items_per_index) {}
  ClassWriter(const ClassWriter&) = delete;
  ClassWriter(ClassWriter&&) = default;

  void Write(const T& item) {
    using namespace io_internal;
    if (HasIndex() && current % items_per_index == 0) {
      inds.push_back(ByteSize());
      if (inds.size() >= kIndexBufferSize) FlushIndex();
    }
    current++;
    size_t sz = item.NumBytes();
    if (!kIsConstSize && kSizeNumberBytes < 8 && sz >= (1ll << (8 * kSizeNumberBytes))) {
      throw std::out_of_range("output size too large");
    }
    size_t old_sz = buf.size();
    buf.resize(old_sz + sz + kSizeNumberBytes);
    if constexpr (!kIsConstSize) {
      uint8_t sz_buf[8] = {};
      IntToBytes<uint64_t>(sz, sz_buf);
      memcpy(buf.data() + old_sz, sz_buf, kSizeNumberBytes);
    }
    item.GetBytes(buf.data() + old_sz + kSizeNumberBytes);
    if (buf.size() >= kBufferSize) Flush();
  }

  void Write(const std::vector<T>& items) {
    for (auto& i : items) Write(i);
  }
};

template <class T>
class ClassReader : public io_internal::ClassReaderImpl<T> {
  using io_internal::ClassReaderImpl<T>::kIsConstSize;
  using io_internal::ClassReaderImpl<T>::kSizeNumberBytes;
  using io_internal::ClassReaderImpl<T>::kBufferSize;
  using io_internal::ClassReaderImpl<T>::ParseBufSize;
  using io_internal::ClassReaderImpl<T>::ReadUntilSize;
  using io_internal::ClassReaderImpl<T>::buf;
  using io_internal::ClassReaderImpl<T>::eof;
  using io_internal::ClassReaderImpl<T>::current;
  using io_internal::ClassReaderImpl<T>::fin;
  using io_internal::ClassReaderImpl<T>::fin_index;
  using io_internal::ClassReaderImpl<T>::items_per_index;

  size_t current_offset;

  uint64_t GetNextSize(size_t buf_size) {
    if constexpr (kIsConstSize) return T::NumBytes();

    ReadUntilSize(current_offset + kSizeNumberBytes, buf_size);
    if (buf.size() < current_offset + kSizeNumberBytes) {
      eof = true;
      throw ReadError("no more elements");
    }
    uint8_t sz_buf[8] = {};
    memcpy(sz_buf, buf.data() + current_offset, kSizeNumberBytes);
    return BytesToInt<uint64_t>(sz_buf);
  }
 public:
  using io_internal::ClassReaderImpl<T>::HasIndex;
  using io_internal::ClassReaderImpl<T>::Position;

  ClassReader(const std::string& fname) :
      io_internal::ClassReaderImpl<T>(fname, !kIsConstSize), current_offset(0) {}
  ClassReader(const ClassReader&) = delete;
  ClassReader(ClassReader&&) = default;

  void SkipOne(size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    uint64_t sz = GetNextSize(buf_size);
    if (current_offset + kSizeNumberBytes + sz > buf.size()) {
      fin.seekg(current_offset + kSizeNumberBytes + sz - buf.size(), std::ios_base::cur);
      buf.clear();
      current_offset = 0;
    } else {
      current_offset += kSizeNumberBytes + sz;
    }
    current++;
  }

  T ReadOne(size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    uint64_t sz = GetNextSize(buf_size);
    ReadUntilSize(current_offset + kSizeNumberBytes + sz, buf_size);
    if (buf.size() < current_offset + kSizeNumberBytes + sz) {
      if constexpr (kIsConstSize) {
        eof = true;
        throw ReadError("no more elements");
      } else {
        throw std::runtime_error("invalid file format");
      }
    }
    T ret(buf.data() + (current_offset + kSizeNumberBytes), sz);
    current_offset += kSizeNumberBytes + sz;
    current++;
    if (current_offset + kSizeNumberBytes >= buf.size()) {
      size_t new_sz = buf.size() - current_offset;
      for (size_t i = 0; i < new_sz; i++) buf[i] = buf[i + current_offset];
      buf.resize(new_sz);
      current_offset = 0;
    }
    return ret;
  }

  std::vector<T> ReadBatch(size_t num, size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    std::vector<T> ret;
    ret.reserve(num);
    try {
      for (size_t i = 0; i < num; i++) ret.push_back(ReadOne(buf_size));
    } catch (ReadError&) {}
    return ret;
  }

  void Seek(size_t location, size_t buf_size = std::string::npos) {
    ParseBufSize(buf_size);
    if constexpr (kIsConstSize) {
      size_t buf_start = current - current_offset / T::NumBytes();
      size_t buf_end = current + (buf.size() - current_offset) / T::NumBytes();
      if (buf_start <= location && location < buf_end) {
        current_offset += ((int64_t)location - current) * T::NumBytes();
      } else {
        buf.clear();
        eof = false;
        fin.clear();
        fin.seekg(T::NumBytes() * location);
        current_offset = 0;
      }
      current = location;
      return;
    }
    if (HasIndex()) {
      size_t target_index_idx = location / items_per_index;
      size_t current_index_idx = current / items_per_index;
      if (target_index_idx != current_index_idx || location < current) {
        buf.clear();
        // +1 because the first element is items_per_index
        fin_index.clear();
        fin_index.seekg((target_index_idx + 1) * sizeof(uint64_t));
        uint8_t sz_buf[16] = {};
        if (!fin_index.read(reinterpret_cast<char*>(sz_buf), 16) || fin_index.gcount() != 16) {
          throw std::out_of_range("index out of range");
        }
        uint64_t start = BytesToInt<uint64_t>(sz_buf), end = BytesToInt<uint64_t>(sz_buf + 8);
        fin.clear();
        fin.seekg(start);
        buf_size = end - start;
        current = target_index_idx * items_per_index;
        current_offset = 0;
      }
    } else if (location < current) {
      buf.clear();
      fin.clear();
      fin.seekg(0);
      current = 0;
      current_offset = 0;
    }
    while (location > current) SkipOne(buf_size);
  }
};
