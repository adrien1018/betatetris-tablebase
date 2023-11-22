#pragma once

#include <cstdlib>

#include "move_search.h"

void InspectBoard(int group, const std::vector<long>& board_idx);
void InspectEdge(int group, const std::vector<long>& board_idx, Level level, int piece);
