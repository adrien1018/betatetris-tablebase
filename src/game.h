#pragma once

#ifdef NO_2KS
constexpr int kLevels = 3;
#else
constexpr int kLevels = 4;
#endif

#ifdef LINE_CAP
constexpr int kLineCap = LINE_CAP;
#else
constexpr int kLineCap = 390;
#ifndef TESTING
#warning "Line cap not defined. Setting to 390."
#endif
#endif

#ifdef DOUBLE_TUCK
constexpr bool kDoubleTuckAllowed = true;
#else
constexpr bool kDoubleTuckAllowed = false;
#endif

#ifdef TETRIS_ONLY
constexpr int kGroups = 10;
#else
constexpr int kGroups = 5;
#endif

enum Level {
  kLevel18,
  kLevel19,
  kLevel29,
  kLevel39
};

alignas(32) constexpr float kTransitionProb[][8] = {
  {1./32, 5./32, 6./32, 5./32, 5./32, 5./32, 5./32}, // T
  {6./32, 1./32, 5./32, 5./32, 5./32, 5./32, 5./32}, // J
  {5./32, 6./32, 1./32, 5./32, 5./32, 5./32, 5./32}, // Z
  {5./32, 5./32, 5./32, 2./32, 5./32, 5./32, 5./32}, // O
  {5./32, 5./32, 5./32, 5./32, 2./32, 5./32, 5./32}, // S
  {6./32, 5./32, 5./32, 5./32, 5./32, 1./32, 5./32}, // L
  {5./32, 5./32, 5./32, 5./32, 6./32, 5./32, 1./32}, // I
};

constexpr int kTransitionProbInt[7][7] = {
// T  J  Z  O  S  L  I (next)
  {1, 5, 6, 5, 5, 5, 5}, // T (current)
  {6, 1, 5, 5, 5, 5, 5}, // J
  {5, 6, 1, 5, 5, 5, 5}, // Z
  {5, 5, 5, 2, 5, 5, 5}, // O
  {5, 5, 5, 5, 2, 5, 5}, // S
  {6, 5, 5, 5, 5, 1, 5}, // L
  {5, 5, 5, 5, 6, 5, 1}, // I
};

constexpr int kLevelSpeedLines[] = {0, 130, 230,
#ifdef NO_2KS
  kLineCap
#else
  330, kLineCap
#endif
};

constexpr int GetLevelByLines(int lines) {
  if (lines < 130) return 18;
  return lines / 10 + 6;
}

constexpr Level GetLevelSpeed(int level) {
  if (level == 18) return kLevel18;
  if (level < 29) return kLevel19;
#ifdef NO_2KS
  return kLevel29;
#else
  if (level < 39) return kLevel29;
  return kLevel39;
#endif
}

constexpr Level GetLevelSpeedByLines(int lines) {
  return GetLevelSpeed(GetLevelByLines(lines));
}

constexpr int Score(int base_lines, int lines) {
#ifdef TETRIS_ONLY
  return base_lines < kLineCap && base_lines + lines >= kLineCap;
#else
  constexpr int kTable[] = {0, 40, 100, 300, 1200};
  return kTable[lines] * (GetLevelByLines(base_lines + lines) + 1);
#endif
}

#ifdef TETRIS_ONLY
constexpr int kGroupInterval = 40;
constexpr int kGroupLineInterval = 4;
constexpr int kCellsMod = 4;
#else
constexpr int kGroupInterval = 10;
constexpr int kGroupLineInterval = 2;
constexpr int kCellsMod = 2;
#endif

constexpr int GetGroupByPieces(int pieces) {
  return pieces * 4 / kGroupLineInterval % kGroups;
}

constexpr int GetGroupByCells(int cells) {
  return cells / kCellsMod % kGroups;
}

constexpr int GetCellsByGroupOffset(int offset, int group) {
  return offset * kGroupInterval + group * kCellsMod;
}

constexpr int NextGroup(int group) {
#ifdef TETRIS_ONLY
  return (group + 1) % kGroups;
#else
  return (group + 2) % kGroups;
#endif
}

// some testcases

static_assert(GetLevelByLines(kLevelSpeedLines[0]) == 18);
static_assert(GetLevelByLines(kLevelSpeedLines[1]) == 19);
static_assert(GetLevelByLines(kLevelSpeedLines[2]) == 29);
#ifndef NO_2KS
static_assert(GetLevelByLines(kLevelSpeedLines[3]) == 39);
#endif

static_assert(GetGroupByPieces(0) == 0);
#ifdef TETRIS_ONLY
static_assert(GetGroupByPieces(1) == 1);
static_assert(GetGroupByPieces(9) == 9);
static_assert(GetGroupByPieces(16) == 6);
#else
static_assert(GetGroupByPieces(1) == 2);
static_assert(GetGroupByPieces(4) == 3);
static_assert(GetGroupByPieces(9) == 3);
#endif
