#pragma once

#include <ostream>
#include "../src/board.h"
#include "../src/position.h"

void PrintTo(const Position& x, std::ostream* os);
void PrintTo(const Board& x, std::ostream* os);
