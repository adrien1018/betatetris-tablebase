#pragma once

#ifdef NO_2KS
constexpr int kLevels = 3;
#else
constexpr int kLevels = 4;
#endif

#ifdef LINE_CAP
constexpr int kLineCap = LINE_CAP;
#else
constexpr int kLineCap = 3300;
#ifndef TESTING
#warning "Line cap not defined. Setting to 3300."
#endif
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
  if (lines < 200) return lines / 10 + 6;
  int offset = (lines - 200) / 10 % 1760; // cycle of 17600 lines
  int base = 26;
  if (offset >= 880) offset -= 10, base += 10;
  int cycle = offset / 290, add = offset % 290;
  return (base + cycle * 210 + std::min(add, 209)) & 255;
}

constexpr Level GetLevelSpeed(int level) {
  if (level <= 18) return kLevel18;
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

constexpr int Score(int lines, int level) {
  constexpr int kTable[] = {0, 40, 100, 300, 1200};
  return kTable[lines] * (level + 1);
}

constexpr int GetGroupByPieces(int pieces) {
  return pieces * 4 / 2 % 5;
}

// some testcases

static_assert(GetLevelByLines(kLevelSpeedLines[0]) == 18);
static_assert(GetLevelByLines(kLevelSpeedLines[1]) == 19);
static_assert(GetLevelByLines(kLevelSpeedLines[2]) == 29);
#ifndef NO_2KS
static_assert(GetLevelByLines(kLevelSpeedLines[3]) == 39);
#endif

static_assert(GetLevelByLines(1310) == 137);
static_assert(GetLevelByLines(2290) == 235);
static_assert(GetLevelByLines(3099) == 235);
static_assert(GetLevelByLines(3300) == 0);
static_assert(GetLevelByLines(3300) == 0);
static_assert(GetLevelByLines(5190) == 189);
static_assert(GetLevelByLines(5999) == 189);

static_assert(GetGroupByPieces(0) == 0);
static_assert(GetGroupByPieces(1) == 2);
static_assert(GetGroupByPieces(4) == 3);
static_assert(GetGroupByPieces(9) == 3);
