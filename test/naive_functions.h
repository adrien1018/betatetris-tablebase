#pragma once

#include <vector>
#include "../board.h"
#include "../move_search.h"

std::vector<ByteBoard> GetPieceMap(const ByteBoard& field, int poly);
ByteBoard PlacePiece(const ByteBoard& b, int poly, int r, int x, int y);
int ClearLines(ByteBoard& field);
PossibleMoves NaiveGetPossibleMoves(const std::vector<ByteBoard>& b, Level level, int adj_frame, const int taps[]);
