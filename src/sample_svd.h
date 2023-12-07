#pragma once

void RunSample(int start_pieces, size_t num_samples, float smooth_pow, size_t seed);
void DoSVD(bool is_ev, float training_split, const std::vector<int>& ranks, size_t seed);
