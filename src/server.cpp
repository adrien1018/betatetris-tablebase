#include "server.h"

#include <array>
#include <boost/asio.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Warray-bounds"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop
#include "game.h"
#include "move.h"
#include "config.h"
#include "tetris.h"
#include "io_hash.h"
#include "board_set.h"

using namespace boost;
using boost::asio::ip::tcp;
using boost::system::error_code;

namespace {

class Connection : public std::enable_shared_from_this<Connection> {
  tcp::socket socket_;
  std::string remote_addr_;
  int remote_port_;

  void Send(const char* buf, size_t len) {
    error_code ec;
    size_t sent = 0;
    while (sent < len) {
      size_t cur = socket_.write_some(asio::buffer(buf + sent, len - sent), ec);
      if (ec) throw std::runtime_error(fmt::format("Write error: {}", ec.message()));
      sent += cur;
    }
  }

  std::vector<uint8_t> ReadUntil(size_t len) {
    std::vector<uint8_t> buf(len);
    error_code ec;
    size_t received = 0;
    while (received < len) {
      size_t cur = socket_.read_some(asio::buffer(buf.data() + received, len - received), ec);
      if (ec) throw std::runtime_error(fmt::format("Read error: {}", ec.message()));
      received += cur;
    }
    return buf;
  }

  Tetris game;
  bool done;
  std::vector<HashMapReader<CompactBoard, BasicIOType<uint32_t>>> board_hash;
  std::vector<CompressedClassReader<NodeMovePositionRange>> move_readers;
  Position prev_pos, prev_placement;
  FrameSequence prev_seq;
  std::array<Position, 7> prev_strats;

  void SendSeq(const FrameSequence& seq) {
    size_t send_size = std::max(1, (int)seq.size());
    std::vector<uint8_t> buf(send_size + 2);
    buf[0] = 0xfe;
    buf[1] = send_size;
    static_assert(sizeof(seq[0]) == 1);
    if (seq.size()) memcpy(buf.data() + 2, seq.data(), seq.size());
    Send(reinterpret_cast<const char*>(buf.data()), buf.size());
  }

  std::array<Position, 7> GetCurrentStrat() {
    auto board = game.GetBoard().ToBytes();
    int group = board.Group();
    auto idx = board_hash[group][board];
    if (!idx) return {Position::Invalid}; // actually {}, since Invalid is (0,0,0)
    size_t move_idx = (size_t)idx.value() * kPieces + game.NowPiece();
    move_readers[group].Seek(move_idx, 0, 0);
    spdlog::debug("Group {}, idx {}, move_idx {}", group, (size_t)idx.value(), move_idx);
    // use 1 to avoid being treated as NULL
    NodeMovePositionRange pos_ranges = move_readers[group].ReadOne(1, 0);
    for (auto& range : pos_ranges.ranges) {
      uint8_t loc = game.GetLines() / 2;
      if (range.start <= loc && loc < range.end) {
        std::string str;
        for (auto& i : range.pos) str += fmt::format("({},{},{})", i.r, i.x, i.y);
        spdlog::debug("{}-{} ({}): {}", range.start, range.end, loc, str);
        return range.pos;
      }
    }
    return {Position::Invalid};
  }

  void StepGame(const Position& pos) {
    if (done) return;
    game.InputPlacement(pos, 0);
    done = game.IsOver();
  }

  void DoPremove() {
    auto strat = GetCurrentStrat();
    if (strat[0] == Position::Invalid) {
      done = true;
      SendSeq({});
      return;
    }
    if (!game.IsNoAdjMove(strat[0])) {
      auto [pos, seq] = game.GetAdjPremove(strat.data());
      prev_pos = pos;
      prev_seq = seq;
      prev_strats = strat;
      StepGame(pos);
      SendSeq(seq);
    } else {
      auto seq = game.GetSequence(strat[0]);
      prev_pos = strat[0];
      prev_seq.clear();
      prev_strats = {Position::Invalid};
      SendSeq(seq);
      SendSeq({});
    }
  }

  void FinishMove(int next_piece) {
    bool prev_none = prev_strats[0] == Position::Invalid;
    if (done) {
      if (!prev_none) SendSeq({});
      return;
    }
    game.SetNextPiece(next_piece);
    if (prev_none) {
      StepGame(prev_pos);
      prev_placement = prev_pos;
    } else {
      Position strat = prev_strats[next_piece];
      size_t old_size = prev_seq.size();
      auto fin_seq = prev_seq;
      game.FinishAdjSequence(fin_seq, prev_pos, strat);
      fin_seq.erase(fin_seq.begin(), fin_seq.begin() + old_size);
      SendSeq(fin_seq);
      StepGame(strat);
      prev_placement = strat;
    }
  }

  void FirstPiece() {
    auto strats = GetCurrentStrat();
    auto pos = strats[game.NextPiece()];
    SendSeq(game.GetSequence(pos));
    if (!game.IsNoAdjMove(strats[0])) {
      StepGame(game.GetAdjPremove(strats.data()).first);
    }
    StepGame(pos);
    prev_placement = pos;
    DoPremove();
  }

  void DoWork() {
    while (true) {
      uint8_t op = ReadUntil(1)[0];
      if (op == 0xff) {
        auto data = ReadUntil(3);
        spdlog::info("New game: {}, {}", (int)data[0], (int)data[1]);
        done = false;
        game.Reset(Board::Ones, 0, data[0], data[1]);
        FirstPiece();
      } else if (op == 0xfd) {
        auto data = ReadUntil(4);
        if (!done && Position{data[0], data[1], data[2]} != prev_placement) {
          spdlog::warn("Error: unexpected placement ({}, {}, {}); expected ({}, {}, {})",
              data[0], data[1], data[2], prev_placement.r, prev_placement.x, prev_placement.y);
          done = true;
        }
        FinishMove(data[3]);
        DoPremove();
      } else {
        throw std::runtime_error("Invalid opcode");
      }
    }
  }
 public:
  Connection(asio::io_context& io_context) :
      socket_(io_context), done(true) {
    for (int i = 0; i < kGroups; i++) {
      board_hash.emplace_back(BoardMapPath(i));
      move_readers.emplace_back(MovePath(i));
    }
  }
  tcp::socket& GetSocket() { return socket_; }
  void Run(const std::string& remote_addr, int remote_port) {
    remote_addr_ = remote_addr;
    remote_port_ = remote_port;
    try {
      DoWork();
    } catch (std::exception& e) {
      spdlog::warn("{}", e.what());
    }
  }
};

class Server {
  boost::asio::io_context& io_context_;
  tcp::acceptor acceptor_;

  void DoAccept() {
    auto new_connection = std::make_shared<Connection>(io_context_);
    acceptor_.async_accept(new_connection->GetSocket(), [this,new_connection](error_code ec) {
      auto& socket = new_connection->GetSocket();
      auto remote_addr = socket.remote_endpoint().address().to_string();
      auto remote_port = socket.remote_endpoint().port();
      if (!ec) {
        spdlog::info("New connection: {}:{}", remote_addr, remote_port);
        std::thread([new_connection,remote_addr,remote_port]() {
          new_connection->Run(remote_addr, remote_port);
        }).detach();
      } else {
        spdlog::info("Accept error: {}", ec.message());
      }
      DoAccept();
    });
  }
 public:
  Server(asio::io_context& io_context, const std::string& addr, int port) :
      io_context_(io_context),
      acceptor_(io_context, tcp::endpoint(asio::ip::make_address(addr), port)) {
    DoAccept();
  }
};

} // namespace

void StartServer(const std::string& bind, int port) {
  asio::io_context io_context;
  Server s(io_context, bind, port);
  io_context.run();
}
