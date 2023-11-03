#pragma once

#include <ostream>
#include "../board.h"
#include "../position.h"

void PrintTo(const Position& x, std::ostream* os);
void PrintTo(const Board& x, std::ostream* os);
