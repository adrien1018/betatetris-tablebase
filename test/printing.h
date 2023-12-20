#pragma once

#include <ostream>
#include "../src/board.h"
#include "../src/position.h"
#include "../src/frame_sequence.h"

void PrintTo(const Position& x, std::ostream* os);
void PrintTo(const Board& x, std::ostream* os);
void PrintTo(const FrameInput& x, std::ostream* os);
void PrintTo(const FrameSequence& x, std::ostream* os);

template <class T, class = decltype(PrintTo(T(), (std::ostream*)nullptr))>
inline std::ostream& operator<<(std::ostream& os, const T& x) {
  PrintTo(x, &os);
  return os;
}
