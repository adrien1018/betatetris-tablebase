#include <thread>
#include <iostream>
#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>

#include "files.h"
#include "config.h"
#include "inspect.h"
#include "board_set.h"

template <class T>
std::vector<T> ParseIntList(const std::string& str) {
  std::vector<T> ret;
  for (size_t i = 0, end; i < str.size(); i = end + 1) {
    end = str.find(',', i);
    if (end == std::string::npos) end = str.size();
    std::string sub = str.substr(i, end - i);
    size_t p0 = sub.find(':');
    if (p0 == std::string::npos) {
      ret.push_back(std::stol(sub));
      continue;
    }
    size_t p1 = sub.find(':', p0 + 1);
    T range_start = std::stol(sub.substr(0, p0));
    T range_end = std::stol(sub.substr(p0 + 1, p1));
    if (p1 == std::string::npos) {
      for (T j = range_start; j < range_end; j++) ret.push_back(j);
      continue;
    }
    T step = std::stol(sub.substr(p1 + 1));
    if (step == 0) continue;
    if (step > 0) {
      for (T j = range_start; j < range_end; j += step) ret.push_back(j);
    } else {
      for (T j = range_start; j > range_end; j += step) ret.push_back(j);
    }
  }
  return ret;
}

Level ParseLevel(int level) {
  if (level == 18) return kLevel18;
  if (level >= 19 && level < 29) return kLevel19;
  if (level >= 29 && level < 39) return kLevel29;
  if (level >= 39) return kLevel39;
  throw std::out_of_range("Invalid level");
}

int ParsePiece(const std::string& str) {
  if (str.size() > 1) throw std::invalid_argument("Invalid piece");
  switch (str[0]) {
    case 'T': return 0;
    case 'J': return 1;
    case 'Z': return 2;
    case 'O': return 3;
    case 'S': return 4;
    case 'L': return 5;
    case 'I': return 6;
    case '0': case '1': case '2': case '3':
    case '4': case '5': case '6': return str[0] - '0';
  }
  throw std::invalid_argument("Invalid piece");
}

int main(int argc, char** argv) {
  spdlog::set_pattern("[%t] %+");
  spdlog::set_level(spdlog::level::debug);

  using namespace argparse;

  ArgumentParser program("main", "1.0");

  auto AddDataDir = [](ArgumentParser& parser) {
    parser.add_argument("data_dir")
      .help("Directory for storing all results");
  };

  ArgumentParser common_opts("common", "", default_arguments::none);
  common_opts.add_argument("-p", "--parallel")
    .help("Number of worker threads to use")
    .metavar("N")
    .default_value((int)std::thread::hardware_concurrency())
    .scan<'i', int>();
  AddDataDir(common_opts);

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
    .metavar("GROUP")
    .default_value("0:5");

  ArgumentParser inspect("inspect", "", default_arguments::help);
  inspect.add_description("Inspect files");

  ArgumentParser inspect_common_opts("inspect_common", "", default_arguments::none);
  inspect_common_opts.add_argument("-g", "--group").required()
    .help("Group (0-4)")
    .metavar("GROUP")
    .scan<'i', int>();
  inspect_common_opts.add_argument("-b", "--board-id").required()
    .help("Board IDs (comma-separated)")
    .metavar("ID");

  ArgumentParser inspect_board_id("board_id", "", default_arguments::help);
  inspect_board_id.add_description("Get board(s) by ID");
  inspect_board_id.add_parents(inspect_common_opts);
  AddDataDir(inspect_board_id);

  ArgumentParser inspect_edge("edge", "", default_arguments::help);
  inspect_edge.add_description("Get edges of a node");
  inspect_edge.add_parents(inspect_common_opts);
  inspect_edge.add_argument("-l", "--level").required()
    .help("Level (18,19,29,39)")
    .metavar("[18/19/29/39]")
    .scan<'i', int>();
  inspect_edge.add_argument("-p", "--piece").required()
    .help("Piece (0-6 or TJZOSLI)")
    .metavar("PIECE");
  AddDataDir(inspect_edge);

  inspect.add_subparser(inspect_board_id);
  inspect.add_subparser(inspect_edge);

  program.add_subparser(preprocess);
  program.add_subparser(build_edges);
  program.add_subparser(inspect);

  try {
    program.parse_args(argc, argv);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    if (program.is_subcommand_used("preprocess")) {
      std::cerr << preprocess;
    } else if (program.is_subcommand_used("build-edges")) {
      std::cerr << build_edges;
    } else if (program.is_subcommand_used("inspect")) {
      auto& subparser = program.at<ArgumentParser>("inspect");
      if (subparser.is_subcommand_used("board_id")) {
        std::cerr << inspect_board_id;
      } else if (subparser.is_subcommand_used("edge")) {
        std::cerr << inspect_edge;
      } else {
        std::cerr << inspect;
      }
    } else {
      std::cerr << program;
    }
    return 1;
  }

  auto SetInspectCommon = [&](const ArgumentParser& args) {
    kDataDir = args.get<std::string>("data_dir");
    int group = args.get<int>("--group");
    auto board_id = ParseIntList<long>(args.get<std::string>("--board-id"));
    return std::make_pair(group, board_id);
  };
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
      auto groups = ParseIntList<int>(args.get<std::string>("--groups"));
      BuildEdges(groups);
    } else if (program.is_subcommand_used("inspect")) {
      auto& subparser = program.at<ArgumentParser>("inspect");
      if (subparser.is_subcommand_used("board_id")) {
        auto& args = subparser.at<ArgumentParser>("board_id");
        auto [group, board_id] = SetInspectCommon(args);
        InspectBoard(group, board_id);
      } else if (subparser.is_subcommand_used("edge")) {
        auto& args = subparser.at<ArgumentParser>("edge");
        auto [group, board_id] = SetInspectCommon(args);
        Level level = ParseLevel(args.get<int>("--level"));
        int piece = ParsePiece(args.get<std::string>("--piece"));
        InspectEdge(group, board_id, level, piece);
      } else {
        std::cerr << inspect;
        return 1;
      }
    } else {
      std::cerr << program;
      return 1;
    }
  } catch (std::logic_error& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
