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
};

class IOTestConstSize : public IOTest {
 protected:
  std::vector<ConstSizeStruct> vec;
  void SetUp(size_t len) {
    gen.seed(0);
    vec.resize(len);
    for (auto& i : vec) {
      for (auto& j : i.arr) j = gen();
    }
    ClassWriter<ConstSizeStruct> writer(kTestFile);
    writer.Write(vec);
  }
};

class IOTestVarSize : public IOTest {
 protected:
  std::vector<VarSizeStruct> vec;
  void SetUp(size_t len, size_t items_per_index = 256, size_t max_elem = 255) {
    gen.seed(0);
    vec.resize(len);
    for (auto& i : vec) {
      i.arr.resize(mrand(0, max_elem)(gen));
      for (auto& j : i.arr) j = gen();
    }
    ClassWriter<VarSizeStruct> writer(kTestFile, items_per_index);
    writer.Write(vec);
  }
};

TEST_F(IOTestConstSize, ReadWrite) {
  SetUp(100);
  {
    ClassReader<ConstSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(100);
    ASSERT_EQ(nvec, vec);
    ASSERT_EQ(0, reader.ReadBatch(1).size());
  }
  {
    ClassReader<ConstSizeStruct> reader(kTestFile);
    auto nvec = reader.ReadBatch(100, 133);
    ASSERT_EQ(nvec, vec);
  }
}

TEST_F(IOTestConstSize, Seek) {
  SetUp(100000);
  ClassReader<ConstSizeStruct> reader(kTestFile);
  for (size_t i = 0; i < 1000; i++) {
    size_t loc = mrand(0, vec.size() - 1)(gen);
    reader.Seek(loc);
    ASSERT_EQ(reader.ReadOne(), vec[loc]);
  }
  for (size_t i = 0; i < 10000; i++) {
    size_t loc = mrand(0, vec.size() - 1)(gen);
    reader.Seek(loc);
    ASSERT_EQ(reader.ReadOne(0), vec[loc]);
  }
}

TEST_F(IOTestVarSize, ReadWrite) {
  for (size_t index : {0, 256}) {
    TearDown();
    SetUp(100, index);
    if (!index) {
      ASSERT_EQ(std::filesystem::is_regular_file(kTestIndexFile), false);
    }
    {
      ClassReader<VarSizeStruct> reader(kTestFile);
      auto nvec = reader.ReadBatch(100);
      ASSERT_EQ(nvec, vec);
      ASSERT_EQ(0, reader.ReadBatch(1).size());
    }
    {
      ClassReader<VarSizeStruct> reader(kTestFile);
      auto nvec = reader.ReadBatch(100, 133);
      ASSERT_EQ(nvec, vec);
    }
  }
}

TEST_F(IOTestVarSize, Seek) {
  for (size_t index : {0, 256}) {
    SetUp(100000, index);
    ClassReader<VarSizeStruct> reader(kTestFile);
    for (size_t i = 0; i < (index ? 1000 : 100); i++) {
      size_t loc = mrand(0, vec.size() - 1)(gen);
      reader.Seek(loc);
      ASSERT_EQ(reader.ReadOne(), vec[loc]);
    }
    for (size_t i = 0; i < (index ? 10000 : 100); i++) {
      size_t loc = mrand(0, vec.size() - 1)(gen);
      reader.Seek(loc);
      ASSERT_EQ(reader.ReadOne(512), vec[loc]);
    }
  }
}

TEST_F(IOTestVarSize, SizeError) {
  EXPECT_THROW({
    SetUp(1000, 256, 1024);
  }, std::out_of_range);
}

} // namespace
