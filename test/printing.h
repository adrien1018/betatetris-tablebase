#pragma once

#include <ostream>
#include "../position.h"

void PrintTo(const Position& x, std::ostream* os) {
    *os << '(' << x.r << ',' << x.x << ',' << x.y << ')';
}
