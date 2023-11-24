#include <gtest/gtest.h>
#include <random>
#include "../src/evaluate.h"

namespace {

using rrand = std::uniform_real_distribution<float>;
  using Vec = std::array<float, 7>;

class NodeEvalTest : public ::testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {}
};

TEST_F(NodeEvalTest, Serialize) {
  std::mt19937_64 gen;
  for (size_t i = 0; i < 100; i++) {
    Vec ev_in, var_in, ev_out, var_out;
    for (auto& j : ev_in) j = rrand(0, 1000)(gen);
    for (auto& j : var_in) j = rrand(0, 1e+14)(gen);
    NodeEval v(ev_in.data(), var_in.data());
    ASSERT_EQ(v.NumBytes(), 56); // 4 * 14
    uint8_t buf[56];
    v.GetBytes(buf);
    NodeEval v2(buf, 56);
    v2.GetEv(ev_out.data());
    v2.GetVar(var_out.data());
    ASSERT_EQ(ev_in, ev_out);
    ASSERT_EQ(var_in, var_out);
  }
}

TEST_F(NodeEvalTest, MaxWith) {
  Vec ev1  = {{0, 1, 0, 1, 0, 0, 0}};
  Vec ev2  = {{1, 0, 1, 0, 1, 1, 1}};
  Vec var1 = {{1, 1, 1, 1, 1, 1, 1}};
  Vec var2 = {{2, 2, 2, 2, 2, 2, 2}};

  Vec correct_ev  = {{1, 1, 1, 1, 1, 1, 1}};
  Vec correct_var = {{2, 1, 2, 1, 2, 2, 2}};
  auto AssertCorrect = [&](const NodeEval& v) {
    Vec out;
    v.GetEv(out.data());
    ASSERT_EQ(out, correct_ev);
    v.GetVar(out.data());
    ASSERT_EQ(out, correct_var);
  };
  {
    NodeEval v1(ev1.data(), var1.data());
    NodeEval v2(ev2.data(), var2.data());
    v1.MaxWith(v2);
    AssertCorrect(v1);
  }
  {
    NodeEval v1(ev1.data(), var1.data());
    NodeEval v2(ev2.data(), var2.data());
    v2.MaxWith(v1);
    AssertCorrect(v2);
  }
}

TEST_F(NodeEvalTest, EvVar1) {
  Vec ev  = {{1, 2, 3, 4, 5, 6, 7}};
  Vec var = {{0, 0, 0, 0, 0, 0, 0}};
  __m256 probs = _mm256_set_ps(0, 1./7, 1./7, 1./7, 1./7, 1./7, 1./7, 1./7); // reverse order

  NodeEval v(ev.data(), var.data());
  float res_ev = v.Dot(probs);
  float res_var = v.DotVar(probs, res_ev);
  ASSERT_EQ(res_ev, 4);
  ASSERT_EQ(res_var, float(7 * 7 - 1) / 12);
}

TEST_F(NodeEvalTest, EvVar2) {
  // Mix of (0,2) random and (1,3) random -> (0,1,2,3) random
  Vec ev  = {{1, 2, 0, 0, 0, 0, 0}};
  Vec var = {{1, 1, 0, 0, 0, 0, 0}};
  __m256 probs = _mm256_set_ps(0, 0, 0, 0, 0, 0, 1./2, 1./2); // reverse order

  NodeEval v(ev.data(), var.data());
  float res_ev = v.Dot(probs);
  float res_var = v.DotVar(probs, res_ev);
  ASSERT_EQ(res_ev, 1.5);
  ASSERT_EQ(res_var, float(4 * 4 - 1) / 12);
}

} // namespace
