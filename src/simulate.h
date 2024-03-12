#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct SimulateResult {
  uint64_t seed;
  int transitions[3];
  int lines, score;
  int end_height;
};

std::vector<SimulateResult> Simulate(const std::vector<uint64_t>& seeds, bool gym_rng);
void OutputSimulate(const std::string& seed_file, const std::string& out_file, bool gym_rng);
