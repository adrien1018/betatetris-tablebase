#pragma once

#include <string>

void StartFCEUXServer(const std::string& bind, int port, bool one_conn);
void StartBoardServer(const std::string& bind, int port, const std::string& threshold_name, bool one_conn);
