#include <thread>
#include <iostream>
#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>

#include "files.h"
#include "config.h"
#include "board_set.h"

std::vector<int> ParseIntList(const std::string& str) {
  std::vector<int> ret;
  for (size_t i = 0, end; i < str.size(); i = end) {
    end = str.find(',', i);
    if (end == std::string::npos) end = str.size();
    std::string sub = str.substr(i, end - i);
    size_t p0 = sub.find(':');
    if (p0 == std::string::npos) {
      ret.push_back(std::stoi(sub));
      continue;
    }
    size_t p1 = sub.find(':', p0);
    int range_start = std::stoi(sub.substr(0, p0));
    int range_end = std::stoi(sub.substr(p0 + 1, p1));
    if (p1 == std::string::npos) {
      for (int j = range_start; j < range_end; j++) ret.push_back(j);
      continue;
    }
    int step = std::stoi(sub.substr(p1 + 1));
    if (step == 0) continue;
    if (step > 0) {
      for (int j = range_start; j < range_end; j += step) ret.push_back(j);
    } else {
      for (int j = range_start; j > range_end; j += step) ret.push_back(j);
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  spdlog::set_pattern("[%t] %+");
  spdlog::set_level(spdlog::level::debug);

  using namespace argparse;

  ArgumentParser program("main", "1.0");

  ArgumentParser common_opts("data_dir", "", default_arguments::none);
  common_opts.add_argument("-p", "--parallel")
    .help("Number of worker threads to use")
    .metavar("N")
    .default_value((int)std::thread::hardware_concurrency())
    .scan<'i', int>();
  common_opts.add_argument("data_dir")
    .help("Directory for storing all results");

  ArgumentParser preprocess("preprocess", "", default_arguments::help);
  preprocess.add_description("Preprocess a board file");
  preprocess.add_parents(common_opts);
  preprocess.add_argument("board_file")
    .help("Board file");

  ArgumentParser build_edges("build-edges", "", default_arguments::help);
  build_edges.add_description("Build edges from boards");
  build_edges.add_parents(common_opts);
  build_edges.add_argument("-g", "--groups")
    .help("The groups to build (0-4, comma-separated, support Python-like range)")
    .default_value("0:5");

  program.add_subparser(preprocess);
  program.add_subparser(build_edges);

  try {
    program.parse_args(argc, argv);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    if (program.is_subcommand_used("preprocess")) {
      std::cerr << preprocess;
    } else if (program.is_subcommand_used("build-edges")) {
      std::cerr << build_edges;
    } else {
      std::cerr << program;
    }
    return 1;
  }

  auto SetCommon = [&](const ArgumentParser& args) {
    kParallel = args.get<int>("-p");
    kDataDir = args.get<std::string>("data_dir");
  };

  try {
    if (program.is_subcommand_used("preprocess")) {
      auto& args = program.at<ArgumentParser>("preprocess");
      SetCommon(args);
      std::filesystem::path board_file = args.get<std::string>("board_file");
      SplitBoards(board_file);
    } else if (program.is_subcommand_used("build-edges")) {
      auto& args = program.at<ArgumentParser>("build-edges");
      SetCommon(args);
      auto groups = ParseIntList(args.get<std::string>("--groups"));
      BuildEdges(groups);
    } else {
      std::cerr << program;
      return 1;
    }
  } catch (std::logic_error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
