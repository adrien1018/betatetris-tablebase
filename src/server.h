#pragma once

#include <string>

void StartFCEUXServer(const std::string& bind, int port);
void StartBoardServer(const std::string& bind, int port, const std::string& threshold_name);
