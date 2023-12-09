#pragma once

void SampleFromEval(
    const std::vector<NodeEval>& val, size_t num_samples, float smooth_pow, size_t seed, uint32_t mark,
    std::vector<std::pair<uint32_t, uint32_t>>& ret);
void RunSample(int start_pieces, size_t num_samples, float smooth_pow, size_t seed);
void DoSVD(bool is_ev, float training_split, const std::vector<int>& ranks, size_t seed);
