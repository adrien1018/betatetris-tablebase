#include <thread>
#include <iostream>
#include <argparse/argparse.hpp>

#include "files.h"
#include "config.h"
#include "board_set.h"

int main(int argc, char** argv) {
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

  ArgumentParser preprocess("preprocess", "", default_arguments::none);
  preprocess.add_description("Preprocess a board file");
  preprocess.add_parents(common_opts);
  preprocess.add_argument("board_file")
    .help("Board file");

  program.add_subparser(preprocess);

  try {
    program.parse_args(argc, argv);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    if (program.is_subcommand_used("preprocess")) {
      std::cerr << preprocess;
    } else {
      std::cerr << program;
    }
    return 1;
  }

  if (program.is_subcommand_used("preprocess")) {
    auto& args = program.at<ArgumentParser>("preprocess");
    kParallel = args.get<int>("-p");
    kDataDir = args.get<std::string>("data_dir");
    std::filesystem::path board_file = args.get<std::string>("board_file");
    SplitBoards(board_file);
  } else {
    std::cerr << program;
    return 1;
  }
}
