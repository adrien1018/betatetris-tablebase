#include <array>
#include <random>
#include <filesystem>
#include <gtest/gtest.h>
#include "../src/io.h"

namespace {

using mrand = std::uniform_int_distribution<size_t>;

struct ConstSizeStruct {
  static constexpr bool kIsConstSize = true;
  std::array<uint8_t, 64> arr;

  bool operator==(const ConstSizeStruct&) const = default;
  static constexpr size_t NumBytes() {
    return 64;
  }
  void GetBytes(uint8_t x[]) const {
    memcpy(x, arr.data(), 64);
  }
  ConstSizeStruct() = default;
  ConstSizeStruct(const uint8_t data[], size_t sz) {
    if (sz != 64) throw 1;
    memcpy(arr.data(), data, 64);
  }
};

struct VarSizeStruct {
  static constexpr bool kIsConstSize = false;
  static constexpr size_t kSizeNumberBytes = 1;
  std::vector<uint8_t> arr;

  bool operator==(const VarSizeStruct&) const = default;
  size_t NumBytes() const {
    return arr.size();
  }
  void GetBytes(uint8_t x[]) const {
    memcpy(x, arr.data(), arr.size());
  }
  VarSizeStruct() = default;
  VarSizeStruct(const uint8_t data[], size_t sz) {
    arr.resize(sz);
    memcpy(arr.data(), data, sz);
  }
};

const std::string kTestFile = "./io-test-file";
const std::string kTestIndexFile = kTestFile + ".index";

class IOTest : public ::testing::Test {
 protected:
  std::mt19937_64 gen;
  void TearDown() override {
    std::filesystem::remove(kTestFile);
    std::filesystem::remove(kTestIndexFile);
  }

  template <class Reader, class T, class... Args>
  void TestSeek(size_t iters, Reader& reader, const T& vec, Args&&... args) {
    for (size_t i = 0; i < iters; i++) {
      size_t num = mrand(0, 3)(gen) ? 1 : std::min((size_t)64, vec.size() - 1);
      size_t loc = mrand(0, vec.size() - num)(gen);
      reader.Seek(loc);
      for (size_t j = 0; j < num; j++) {
        auto read = reader.ReadOne(std::forward<Args>(args)...);
        ASSERT_EQ(read, vec[loc + j]);
      }
    }
  }
};

class IOTestConstSize : public IOTest {
 protected:
  std::vector<ConstSizeStruct> vec;
  void SetUp(size_t len, bool compressed = false, size_t items_per_index = 64) {
    gen.seed(0);
    vec.resize(len);
    for (auto& i : vec) {
      for (auto& j : i.arr) j = gen() % 16;
    }
    if (compressed) {
      CompressedClassWriter<ConstSizeStruct> writer(kTestFile, items_per_index);
      writer.Write(vec);
    } else {
      ClassWriter<ConstSizeStruct> writer(kTestFile);
      writer.Write(vec);
    }
  }
};

class IOTestVarSize : public IOTest {
 protected:
  std::vector<VarSizeStruct> vec;
  void SetUp(size_t len, size_t items_per_index, bool compressed = false, size_t max_elem = 255) {
    gen.seed(0);
    vec.resize(len);
    for (auto& i : vec) {
      i.arr.resize(mrand(0, max_elem)(gen));
      for (auto& j : i.arr) j = gen() % 16;
    }
    if (compressed) {
      CompressedClassWriter<VarSizeStruct> writer(kTestFile, items_per_index);
      writer.Write(vec);
    } else {
      ClassWriter<VarSizeStruct> writer(kTestFile, items_per_index);
      writer.Write(vec);
    }
  }
};

TEST_F(IOTestConstSize, ReadWrite) {
  SetUp(1000);
  ASSERT_EQ(std::filesystem::is_regular_file(kTestIndexFile), false);
  {
    ClassReader<ConstSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(1000);
    ASSERT_EQ(nvec, vec);
    ASSERT_EQ(0, reader.ReadBatch(1).size());
  }
  {
    ClassReader<ConstSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(1000, 133);
    ASSERT_EQ(nvec, vec);
  }
}

TEST_F(IOTestConstSize, ReadWriteCompressed) {
  SetUp(1000, true);
  ASSERT_EQ(std::filesystem::is_regular_file(kTestIndexFile), true);
  {
    CompressedClassReader<ConstSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(1000);
    ASSERT_EQ(nvec, vec);
    ASSERT_EQ(0, reader.ReadBatch(1).size());
  }
  {
    CompressedClassReader<ConstSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(1000, 133, 144);
    ASSERT_EQ(nvec, vec);
  }
}

TEST_F(IOTestConstSize, Seek) {
  SetUp(100000);
  ClassReader<ConstSizeStruct> reader(kTestFile);
  TestSeek(1000, reader, vec);
  TestSeek(10000, reader, vec, 0);
}

TEST_F(IOTestConstSize, SeekCompressed) {
  SetUp(100000, true);
  CompressedClassReader<ConstSizeStruct> reader(kTestFile);
  TestSeek(1000, reader, vec);
  TestSeek(10000, reader, vec, 0, 0);
}

TEST_F(IOTestVarSize, ReadWrite) {
  for (size_t index : {0, 256}) {
    TearDown();
    SetUp(1000, index);
    ASSERT_EQ(std::filesystem::is_regular_file(kTestIndexFile), (bool)index);
    {
      ClassReader<VarSizeStruct> reader(kTestFile);
      auto nvec = reader.ReadBatch(1000);
      ASSERT_EQ(nvec, vec);
      ASSERT_EQ(0, reader.ReadBatch(1).size());
    }
    {
      ClassReader<VarSizeStruct> reader(kTestFile);
      auto nvec = reader.ReadBatch(1000, 133);
      ASSERT_EQ(nvec, vec);
    }
  }
}

TEST_F(IOTestVarSize, ReadWriteCompressed) {
  SetUp(1000, 256, true);
  ASSERT_EQ(std::filesystem::is_regular_file(kTestIndexFile), true);
  {
    CompressedClassReader<VarSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(1000);
    ASSERT_EQ(nvec, vec);
    ASSERT_EQ(0, reader.ReadBatch(1).size());
  }
  {
    CompressedClassReader<VarSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(1000, 133, 144);
    ASSERT_EQ(nvec, vec);
  }
}

TEST_F(IOTestVarSize, Seek) {
  for (size_t index : {0, 256}) {
    SetUp(100000, index);
    ClassReader<VarSizeStruct> reader(kTestFile);
    TestSeek(index ? 1000 : 100, reader, vec);
    TestSeek(index ? 10000 : 100, reader, vec, 512);
  }
}

TEST_F(IOTestVarSize, SeekCompressed) {
  SetUp(100000, 64, true);
  CompressedClassReader<VarSizeStruct> reader(kTestFile);
  TestSeek(1000, reader, vec);
  TestSeek(10000, reader, vec, 512, 512);
}

TEST_F(IOTestVarSize, SizeError) {
  EXPECT_THROW({
    SetUp(1000, 256, false, 1024);
  }, std::out_of_range);
}

} // namespace
