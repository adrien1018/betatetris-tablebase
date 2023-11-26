#pragma once

#include <cstdlib>

#include "move_search.h"

void InspectBoard(int group, const std::vector<long>& board_idx);
void InspectBoardStats(int group);
void InspectEdge(int group, const std::vector<long>& board_idx, Level level, int piece);
void InspectEdgeStats(int group, Level level);
void InspectValue(int pieces, const std::vector<long>& board_idx);
