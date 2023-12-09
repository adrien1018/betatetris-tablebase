#pragma once

#include <filesystem>

void SampleTrainingBoards(
    const std::vector<int>& start_pieces_group, size_t num_samples, float smooth_pow, size_t seed,
    const std::filesystem::path& output_path);
