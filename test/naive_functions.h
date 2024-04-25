#pragma once

#include <array>
#include <vector>
#include "../src/board.h"
#include "../src/move_search.h"

std::vector<ByteBoard> GetPieceMap(const ByteBoard& field, int poly);
ByteBoard PlacePiece(const ByteBoard& b, int poly, int r, int x, int y);
int ClearLines(ByteBoard& field);
PossibleMoves NaiveGetPossibleMoves(const std::vector<ByteBoard>& b, Level level, int adj_frame, const int taps[]);

ByteBoard NaiveNoroPossibleMoves(const ByteBoard& b, int inputs_per_row, bool do_tuck);

int NumOverhang(const ByteBoard& b);
std::array<int, 10> ColumnHeights(const ByteBoard& b);
