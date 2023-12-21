#include <thread>
#include <iostream>
#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>

#include "move.h"
#include "files.h"
#include "config.h"
#include "inspect.h"
#include "evaluate.h"
#include "board_set.h"
#include "sample_svd.h"
#include "sample_train.h"

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

  auto DataDirArg = [](ArgumentParser& parser) {
    parser.add_argument("data_dir")
      .help("Directory for storing all results");
  };
  auto ParallelArg = [](ArgumentParser& parser) {
    parser.add_argument("-p", "--parallel")
      .help("Number of worker threads to use")
      .metavar("N")
      .default_value((int)std::thread::hardware_concurrency())
      .scan<'i', int>();
  };
  auto GroupArg = [](ArgumentParser& parser) {
    parser.add_argument("-g", "--group").required()
      .help("Group (0-4)")
      .metavar("GROUP")
      .scan<'i', int>();
  };
  auto BoardIDArg = [](ArgumentParser& parser) {
    parser.add_argument("-b", "--board-id").required()
      .help("Board IDs (comma-separated)")
      .metavar("ID");
  };
  auto LevelArg = [](ArgumentParser& parser) {
    parser.add_argument("-l", "--level").required()
      .help("Level (18,19,29,39)")
      .metavar("[18/19/29/39]")
      .scan<'i', int>();
  };
  auto IOThreadsArg = [](ArgumentParser& parser) {
    parser.add_argument("-i", "--io-threads")
      .help("Number of readers")
      .metavar("N")
      .scan<'i', int>()
      .default_value(4);
  };
  auto ResumeArg = [](ArgumentParser& parser) {
    parser.add_argument("-r", "--resume")
      .help("Resume from a previous checkpoint")
      .metavar("PIECES")
      .scan<'i', int>()
      .default_value(-1);
  };
  auto NumSamplesArg = [](ArgumentParser& parser) {
    parser.add_argument("-n", "--num-samples-per-group").required()
      .help("Number of samples for each group")
      .scan<'i', long>();
  };
  auto PowArg = [](ArgumentParser& parser) {
    parser.add_argument("--pow")
      .help("Exponent used for scaling to get more high-valued samples")
      .scan<'f', float>()
      .default_value(0.5f);
  };
  auto SeedArg = [](ArgumentParser& parser) {
    parser.add_argument("--seed")
      .help("Random seed")
      .scan<'i', long>()
      .default_value(0l);
  };

  ArgumentParser preprocess("preprocess", "", default_arguments::help);
  preprocess.add_description("Preprocess a board file");
  DataDirArg(preprocess);
  ParallelArg(preprocess);
  preprocess.add_argument("board_file")
    .help("Board file");

  ArgumentParser build_edges("build-edges", "", default_arguments::help);
  build_edges.add_description("Build edges from boards");
  DataDirArg(build_edges);
  ParallelArg(build_edges);
  build_edges.add_argument("-g", "--groups")
    .help("The groups to build (0-4, comma-separated, support Python-like range)")
    .metavar("GROUP")
    .default_value("0:5");

  ArgumentParser evaluate("evaluate", "", default_arguments::help);
  evaluate.add_description("Calculate values of every board");
  DataDirArg(evaluate);
  ParallelArg(evaluate);
  IOThreadsArg(evaluate);
  ResumeArg(evaluate);
  evaluate.add_argument("-c", "--checkpoints").required()
    .help("Checkpoints (in pieces) to save the evaluate result (comma-separated, support Python-like range)");
  evaluate.add_argument("-s", "--store-sample")
    .help("Store sampled values (must have sample file available)")
    .default_value(false)
    .implicit_value(true);

  ArgumentParser move_cal("move", "", default_arguments::help);
  move_cal.add_description("Calculate moves of every board");
  DataDirArg(move_cal);
  ParallelArg(move_cal);
  IOThreadsArg(move_cal);
  ResumeArg(move_cal);
  move_cal.add_argument("-e", "--until")
    .help("Calculate until this many pieces")
    .metavar("PIECES")
    .scan<'i', int>()
    .default_value(0);

  ArgumentParser move_merge("move-merge", "", default_arguments::help);
  move_merge.add_description("Merge moves");
  DataDirArg(move_merge);
  ParallelArg(move_merge);
  move_merge.add_argument("-s", "--start").required()
    .help("Start pieces")
    .metavar("PIECES")
    .scan<'i', int>()
    .default_value(-1);
  move_merge.add_argument("-e", "--end").required()
    .help("End pieces")
    .metavar("PIECES")
    .scan<'i', int>()
    .default_value(-1);
  move_merge.add_argument("-d", "--delete")
    .help("Delete files after merge")
    .default_value(false)
    .implicit_value(true);
  move_merge.add_argument("-w", "--whole")
    .help("Whole merge")
    .default_value(false)
    .implicit_value(true);

  ArgumentParser sample_svd("sample-svd", "", default_arguments::help);
  sample_svd.add_description("Sample boards for SVD");
  DataDirArg(sample_svd);
  ParallelArg(sample_svd);
  IOThreadsArg(sample_svd);
  sample_svd.add_argument("-s", "--start-pieces").required()
    .help("Start pieces for sampling (must be a saved evaluate result)")
    .metavar("PIECES")
    .scan<'i', int>();
  NumSamplesArg(sample_svd);
  PowArg(sample_svd);
  SeedArg(sample_svd);

  ArgumentParser svd("svd", "", default_arguments::help);
  svd.add_description("Run SVD for value compression");
  svd.add_argument("-r", "--ranks")
    .help("The ranks to build (comma-separated, support Python-like range)")
    .default_value("1:65");
  svd.add_argument("-t", "--training-split")
    .help("Portion of training data")
    .scan<'f', float>()
    .default_value(0.5f);
  SeedArg(svd);
  svd.add_argument("ev_var").required()
    .help("Value type (ev/var)")
    .metavar("ev/var");
  DataDirArg(svd);

  ArgumentParser sample_train("sample-train", "", default_arguments::help);
  sample_train.add_description("Sample boards for training");
  DataDirArg(sample_train);
  ParallelArg(sample_train);
  IOThreadsArg(sample_train);
  sample_train.add_argument("-s", "--start-pieces").required()
    .help("Start pieces for sampling (must be a saved evaluate result; comma-separated)");
  sample_train.add_argument("-o", "--output").required()
    .help("Output file");
  sample_train.add_argument("-z", "--zero-ratio")
    .help("Portion of samples sampled from ev=0 nodes")
    .scan<'f', float>()
    .default_value(0.5f);
  sample_train.add_argument("-h", "--zero-high-ratio")
    .help("Portion of ev=0 samples sampled from nodes with the largest other-piece ev")
    .scan<'f', float>()
    .default_value(0.4f);
  NumSamplesArg(sample_train);
  PowArg(sample_train);
  SeedArg(sample_train);

  ArgumentParser inspect("inspect", "", default_arguments::help);
  inspect.add_description("Inspect files");

  ArgumentParser inspect_board_id("board-id", "", default_arguments::help);
  inspect_board_id.add_description("Get board(s) by ID");
  GroupArg(inspect_board_id);
  BoardIDArg(inspect_board_id);
  DataDirArg(inspect_board_id);

  ArgumentParser inspect_board_stats("board-stats", "", default_arguments::help);
  inspect_board_stats.add_description("Get board stats");
  GroupArg(inspect_board_stats);
  DataDirArg(inspect_board_stats);

  ArgumentParser inspect_edge("edge", "", default_arguments::help);
  inspect_edge.add_description("Get edges of a node");
  GroupArg(inspect_edge);
  BoardIDArg(inspect_edge);
  LevelArg(inspect_edge);
  inspect_edge.add_argument("-p", "--piece").required()
    .help("Piece (0-6 or TJZOSLI)")
    .metavar("PIECE");
  DataDirArg(inspect_edge);

  ArgumentParser inspect_edge_stats("edge-stats", "", default_arguments::help);
  inspect_edge_stats.add_description("Get edge stats");
  GroupArg(inspect_edge_stats);
  LevelArg(inspect_edge_stats);
  DataDirArg(inspect_edge_stats);

  ArgumentParser inspect_value("value", "", default_arguments::help);
  inspect_value.add_description("Get values of a node");
  BoardIDArg(inspect_value);
  inspect_value.add_argument("-p", "--pieces").required()
    .help("Location (pieces)")
    .metavar("PIECES")
    .scan<'i', int>();
  DataDirArg(inspect_value);

  inspect.add_subparser(inspect_board_id);
  inspect.add_subparser(inspect_board_stats);
  inspect.add_subparser(inspect_edge);
  inspect.add_subparser(inspect_edge_stats);
  inspect.add_subparser(inspect_value);

  program.add_subparser(preprocess);
  program.add_subparser(build_edges);
  program.add_subparser(evaluate);
  program.add_subparser(move_cal);
  program.add_subparser(move_merge);
  program.add_subparser(sample_svd);
  program.add_subparser(svd);
  program.add_subparser(sample_train);
  program.add_subparser(inspect);

  try {
    program.parse_args(argc, argv);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    if (program.is_subcommand_used("preprocess")) {
      std::cerr << preprocess;
    } else if (program.is_subcommand_used("build-edges")) {
      std::cerr << build_edges;
    } else if (program.is_subcommand_used("evaluate")) {
      std::cerr << evaluate;
    } else if (program.is_subcommand_used("move")) {
      std::cerr << move_cal;
    } else if (program.is_subcommand_used("move-merge")) {
      std::cerr << move_merge;
    } else if (program.is_subcommand_used("sample-svd")) {
      std::cerr << sample_svd;
    } else if (program.is_subcommand_used("svd")) {
      std::cerr << svd;
    } else if (program.is_subcommand_used("sample-train")) {
      std::cerr << sample_train;
    } else if (program.is_subcommand_used("inspect")) {
      auto& subparser = program.at<ArgumentParser>("inspect");
      if (subparser.is_subcommand_used("board-id")) {
        std::cerr << inspect_board_id;
      } else if (subparser.is_subcommand_used("board-stats")) {
        std::cerr << inspect_board_stats;
      } else if (subparser.is_subcommand_used("edge")) {
        std::cerr << inspect_edge;
      } else if (subparser.is_subcommand_used("edge-stats")) {
        std::cerr << inspect_edge_stats;
      } else if (subparser.is_subcommand_used("value")) {
        std::cerr << inspect_value;
      } else {
        std::cerr << inspect;
      }
    } else {
      std::cerr << program;
    }
    return 1;
  }

  auto SetParallel = [&](const ArgumentParser& args) {
    kParallel = args.get<int>("-p");
  };
  auto SetDataDir = [&](const ArgumentParser& args) {
    kDataDir = args.get<std::string>("data_dir");
  };
  auto SetIOThreads = [&](const ArgumentParser& args) {
    kIOThreads = args.get<int>("--io-threads");
  };
  auto GetGroup = [](const ArgumentParser& args) {
    return args.get<int>("--group");
  };
  auto GetResume = [](const ArgumentParser& args) {
    return args.get<int>("--resume");
  };
  auto GetBoardID = [](const ArgumentParser& args) {
    return ParseIntList<long>(args.get<std::string>("--board-id"));
  };
  auto GetLevel = [](const ArgumentParser& args) {
    return ParseLevel(args.get<int>("--level"));
  };
  auto GetNumSamples = [](const ArgumentParser& args) {
    return args.get<long>("--num-samples-per-group");
  };
  auto GetPow = [](const ArgumentParser& args) {
    return args.get<float>("--pow");
  };
  auto GetSeed = [](const ArgumentParser& args) {
    return args.get<long>("--seed");
  };

  try {
    if (program.is_subcommand_used("preprocess")) {
      auto& args = program.at<ArgumentParser>("preprocess");
      SetParallel(args);
      SetDataDir(args);
      std::filesystem::path board_file = args.get<std::string>("board_file");
      SplitBoards(board_file);
    } else if (program.is_subcommand_used("build-edges")) {
      auto& args = program.at<ArgumentParser>("build-edges");
      SetParallel(args);
      SetDataDir(args);
      auto groups = ParseIntList<int>(args.get<std::string>("--groups"));
      BuildEdges(groups);
    } else if (program.is_subcommand_used("evaluate")) {
      auto& args = program.at<ArgumentParser>("evaluate");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int resume = GetResume(args);
      auto checkpoints = ParseIntList<int>(args.get<std::string>("--checkpoints"));
      bool sample = args.get<bool>("--store-sample");
      RunEvaluate(resume, checkpoints, sample);
    } else if (program.is_subcommand_used("move")) {
      auto& args = program.at<ArgumentParser>("move");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int resume = GetResume(args);
      int until = args.get<int>("--until");
      RunCalculateMoves(resume, until);
    } else if (program.is_subcommand_used("move-merge")) {
      auto& args = program.at<ArgumentParser>("move-merge");
      SetParallel(args);
      SetDataDir(args);
      int start = args.get<int>("--start");
      int end = args.get<int>("--end");
      bool whole = args.get<bool>("--whole");
      bool delete_after = args.get<bool>("--delete");
      if (whole) {
        throw std::runtime_error("not implemented");
      } else if (start != -1 || end != -1) {
        MergeRanges(start, end, delete_after);
      } else {
        throw std::runtime_error("start / end not given");
      }
    } else if (program.is_subcommand_used("sample-svd")) {
      auto& args = program.at<ArgumentParser>("sample-svd");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      int pieces = args.get<int>("--start-pieces");
      long num_samples = GetNumSamples(args);
      float smooth_pow = GetPow(args);
      long seed = GetSeed(args);
      RunSample(pieces, num_samples, smooth_pow, seed);
    } else if (program.is_subcommand_used("svd")) {
      auto& args = program.at<ArgumentParser>("svd");
      SetDataDir(args);
      auto ranks = ParseIntList<int>(args.get<std::string>("--ranks"));
      float training_split = args.get<float>("--training-split");
      long seed = GetSeed(args);
      bool is_ev = args.get<std::string>("ev_var") == "ev";
      DoSVD(is_ev, training_split, ranks, seed);
    } else if (program.is_subcommand_used("sample-train")) {
      auto& args = program.at<ArgumentParser>("sample-train");
      SetParallel(args);
      SetDataDir(args);
      SetIOThreads(args);
      auto pieces = ParseIntList<int>(args.get<std::string>("--start-pieces"));
      long num_samples = GetNumSamples(args);
      float smooth_pow = GetPow(args);
      float zero_ratio = args.get<float>("--zero-ratio");
      float zero_high_ratio = args.get<float>("--zero-high-ratio");
      long seed = GetSeed(args);
      std::filesystem::path output = args.get<std::string>("--output");
      SampleTrainingBoards(pieces, num_samples, zero_ratio, zero_high_ratio, smooth_pow, seed, output);
    } else if (program.is_subcommand_used("inspect")) {
      auto& subparser = program.at<ArgumentParser>("inspect");
      if (subparser.is_subcommand_used("board-id")) {
        auto& args = subparser.at<ArgumentParser>("board-id");
        auto group = GetGroup(args);
        auto board_id = GetBoardID(args);
        SetDataDir(args);
        InspectBoard(group, board_id);
      } else if (subparser.is_subcommand_used("board-stats")) {
        auto& args = subparser.at<ArgumentParser>("board-stats");
        auto group = GetGroup(args);
        SetDataDir(args);
        InspectBoardStats(group);
      } else if (subparser.is_subcommand_used("edge")) {
        auto& args = subparser.at<ArgumentParser>("edge");
        auto group = GetGroup(args);
        auto board_id = GetBoardID(args);
        Level level = GetLevel(args);
        int piece = ParsePiece(args.get<std::string>("--piece"));
        SetDataDir(args);
        InspectEdge(group, board_id, level, piece);
      } else if (subparser.is_subcommand_used("edge-stats")) {
        auto& args = subparser.at<ArgumentParser>("edge-stats");
        auto group = GetGroup(args);
        Level level = GetLevel(args);
        SetDataDir(args);
        InspectEdgeStats(group, level);
      } else if (subparser.is_subcommand_used("value")) {
        auto& args = subparser.at<ArgumentParser>("value");
        int pieces = args.get<int>("--pieces");
        auto board_id = GetBoardID(args);
        SetDataDir(args);
        InspectValue(pieces, board_id);
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
