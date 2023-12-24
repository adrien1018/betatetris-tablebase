#include <random>
#include <gtest/gtest.h>
#include "../src/move.h"

namespace {

using mrand = std::uniform_int_distribution<size_t>;
using rrand = std::uniform_real_distribution<float>;

class MoveTest : public ::testing::Test {
 private:
  std::mt19937_64 gen;
  std::vector<size_t> GenRanges() {
    size_t nums = mrand(1, 6)(gen);
    std::vector<size_t> vec(nums + 1);
    for (auto& i : vec) i = mrand(0, 255 - nums)(gen);
    std::sort(vec.begin(), vec.end());
    for (size_t i = 0; i <= nums; i++) vec[i] += i;
    return vec;
  }
 protected:
  NodeMoveIndexRange GenRandIndexRange() {
    NodeMoveIndexRange ret;
    auto ranges = GenRanges();
    for (size_t i = 0; i < ranges.size() - 1; i++) {
      NodeMoveIndex item;
      for (auto& j : item) j = mrand(0, 20)(gen);
      ret.ranges.push_back({(uint8_t)ranges[i], (uint8_t)ranges[i+1], item});
    }
    return ret;
  }
  NodeMovePositionRange GenRandPositionRange() {
    NodeMovePositionRange ret;
    auto ranges = GenRanges();
    for (size_t i = 0; i < ranges.size() - 1; i++) {
      std::array<Position, 7> item;
      for (auto& j : item) {
        j.r = mrand(0, 3)(gen);
        j.x = mrand(0, 19)(gen);
        j.y = mrand(0, 9)(gen);
      }
      ret.ranges.push_back({(uint8_t)ranges[i], (uint8_t)ranges[i+1], item});
    }
    return ret;
  }
  void SetUp() {}
  void TearDown() override {}
};

NodeMoveIndex GenNodeMoveIndex(int x) {
  NodeMoveIndex a{};
  a[0] = x;
  return a;
}

TEST_F(MoveTest, ModeIndexConstruct) {
  std::array<NodeMoveIndex, 10> inds{};
  {
    NodeMoveIndexRange r(inds.begin(), inds.end(), 10);
    ASSERT_EQ(r.ranges.size(), 1);
    ASSERT_EQ(r.ranges[0].start, 10);
    ASSERT_EQ(r.ranges[0].end, 20);
    ASSERT_EQ(r.ranges[0].idx, inds[0]);
  } {
    inds[0][0] = 1;
    inds[9][0] = 1;
    NodeMoveIndexRange r(inds.begin(), inds.end(), 5);
    ASSERT_EQ(r.ranges.size(), 3);
    std::vector<MoveIndexRange> expected = {
      {5, 6, inds[0]},
      {6, 14, inds[1]},
      {14, 15, inds[9]}
    };
    ASSERT_EQ(r.ranges, expected);
  }
}

TEST_F(MoveTest, ModeIndexMerge) {
  {
    NodeMoveIndexRange r1, r2;
    r1.ranges.push_back({10, 20, GenNodeMoveIndex(1)});
    r1.ranges.push_back({20, 30, GenNodeMoveIndex(2)});
    r2.ranges.push_back({30, 40, GenNodeMoveIndex(2)});
    r2.ranges.push_back({40, 50, GenNodeMoveIndex(3)});
    r1 <<= r2;
    ASSERT_EQ(r1.ranges.size(), 3);
    r2.ranges[0].start = 20;
    ASSERT_EQ(r1.ranges[1], r2.ranges[0]);
    ASSERT_EQ(r1.ranges[2], r2.ranges[1]);
  } {
    NodeMoveIndexRange r1, r2;
    r1.ranges.push_back({10, 20, GenNodeMoveIndex(1)});
    r1.ranges.push_back({20, 30, GenNodeMoveIndex(2)});
    r2.ranges.push_back({30, 40, GenNodeMoveIndex(3)});
    r2.ranges.push_back({40, 50, GenNodeMoveIndex(4)});
    r1 <<= r2;
    ASSERT_EQ(r1.ranges.size(), 4);
    ASSERT_EQ(r1.ranges[2], r2.ranges[0]);
    ASSERT_EQ(r1.ranges[3], r2.ranges[1]);
    EXPECT_ANY_THROW({ r1 <<= r2; });
  }
}

TEST_F(MoveTest, MoveIndexSerialize) {
  for (size_t i = 0; i < 1000; i++) {
    NodeMoveIndexRange a = GenRandIndexRange();
    std::vector<uint8_t> buf(a.NumBytes());
    a.GetBytes(buf.data());
    NodeMoveIndexRange b(buf.data(), buf.size());
    ASSERT_EQ(a.ranges, b.ranges);
  }
}

TEST_F(MoveTest, ModePositionMerge) {
  NodeMovePositionRange r;
  r <<= MovePositionRange{10, 20, {{{0, 0, 0}}}};
  ASSERT_EQ(r.ranges.size(), 1);
  r <<= MovePositionRange{20, 30, {{{0, 1, 0}}}};
  ASSERT_EQ(r.ranges.size(), 2);
  r <<= MovePositionRange{30, 40, {{{0, 1, 0}}}};
  ASSERT_EQ(r.ranges.size(), 2);
  ASSERT_EQ(r.ranges.back().start, 20);
  ASSERT_EQ(r.ranges.back().end, 40);
  r <<= MovePositionRange{40, 50, {{{0, 2, 0}}}};
  EXPECT_ANY_THROW({ r <<= r.ranges.back(); });
}

TEST_F(MoveTest, MovePositionSerialize) {
  for (size_t i = 0; i < 1000; i++) {
    NodeMovePositionRange a = GenRandPositionRange();
    std::vector<uint8_t> buf(a.NumBytes());
    a.GetBytes(buf.data());
    NodeMovePositionRange b(buf.data(), buf.size());
    ASSERT_EQ(a.ranges, b.ranges);
  }
}

} // namespace
