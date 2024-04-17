#pragma once

std::vector<uint8_t> SampleFromEval(
    const std::vector<MoveEval>& val, size_t num_samples, float smooth_pow, size_t seed);
void SampleMaskToIdx(const std::vector<uint8_t>& mask, std::vector<std::pair<uint32_t, uint32_t>>& ret, uint32_t mark);
void RunSample(int start_pieces, size_t num_samples, float smooth_pow, size_t seed);
void DoSVD(bool is_ev, float training_split, const std::vector<int>& ranks, size_t seed);
