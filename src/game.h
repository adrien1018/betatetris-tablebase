#pragma once

constexpr int kLevels = 4;
#ifdef LINE_CAP
constexpr int kLineCap = LINE_CAP;
#else
constexpr int kLineCap = 390;
#ifndef TESTING
#warning "Line cap not defined. Setting to 390."
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

constexpr int kLevelSpeedLines[] = {0, 130, 230, 330};

constexpr int GetLevelByLines(int lines) {
  if (lines < 130) return 18;
  return lines / 10 + 6;
}
static_assert(GetLevelByLines(kLevelSpeedLines[0]) == 18);
static_assert(GetLevelByLines(kLevelSpeedLines[1]) == 19);
static_assert(GetLevelByLines(kLevelSpeedLines[2]) == 29);
static_assert(GetLevelByLines(kLevelSpeedLines[3]) == 39);

constexpr Level GetLevelSpeed(int level) {
  if (level == 18) return kLevel18;
  if (level < 29) return kLevel19;
  if (level < 39) return kLevel29;
  return kLevel39;
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
static_assert(GetGroupByPieces(0) == 0);
static_assert(GetGroupByPieces(1) == 2);
static_assert(GetGroupByPieces(4) == 3);
static_assert(GetGroupByPieces(9) == 3);
