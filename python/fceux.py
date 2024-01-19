#!/usr/bin/env python3

import sys, os.path, socketserver, argparse, re, socket, pickle
import numpy as np, torch

import tetris
from model import Model, obs_to_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
board_cache = {}

class GameConn(socketserver.BaseRequestHandler):
    def read_until(self, sz):
        data = b''
        while len(data) < sz:
            n = self.request.recv(sz - len(data))
            if len(n) == 0:
                raise ConnectionResetError()
            data += n
        return data

    @staticmethod
    def gen_seq(seq):
        if len(seq) == 0: return bytes([0xfe, 0, 1, 0])
        return bytes([0xfe, len(seq) // 256, len(seq) % 256]) + seq.tobytes()

    @staticmethod
    def seq_to_str(seq):
        ret = ''
        for i in seq:
            s = ''
            for val, ch in zip([1, 2, 4, 8], 'LRAB'):
                if (i & val) == val: s += ch
            if s == '': s = '-'
            ret += s + ' '
        return ret.strip()

    def get_strat(self):
        agent_mode = self.game.GetAgentMode()
        with torch.no_grad():
            pi, v = self.models[agent_mode](obs_to_torch(self.game.GetState()), pi_only=True)
            action = torch.argmax(pi, 1).item()
            return (action // 200, action // 10 % 20, action % 10), v[0].item()

    def print_status(self, strats, is_table, data):
        states = [is_table, self.game.GetAgentMode(), self.game.GetRealPieces(), self.game.GetRealLines(), self.game.GetRealLevel(), self.game.GetRunScore(), self.game.GetNowPiece()]
        def StratState(i):
            ntxt = f'{strats[i][0]} {strats[i][1]} {strats[i][2]}'
            if not is_table:
                ntxt += f' {data[i]}'
            return ntxt
        for i in range(7):
            states.append(StratState(i))
        states.append(data if is_table else '')
        print(','.join([str(i) for i in states]), flush=True)

    def get_adj_strat(self, pos):
        with torch.no_grad():
            # adj is disabled in single & pushdown mode
            pi, v = self.models[0](obs_to_torch(self.game.GetAdjStates(*pos)), pi_only=True)
            actions = torch.argmax(pi, 1).flatten().cpu().tolist()
            strats = [(action // 200, action // 10 % 20, action % 10) for action in actions]
            self.print_status(strats, False, v[0].flatten().cpu().tolist())
            return strats

    def query_tablebase(self):
        global board_cache
        query = (
            bytes([1]) +
            self.game.GetBoard().GetBytes() +
            bytes([self.game.GetNowPiece(), self.game.GetStateLines() % 256, self.game.GetStateLines() // 256])
        )
        if query in board_cache:
            result = board_cache[query]
        else:
            self.board_conn.sendall(query)
            result = self.board_conn.recv(22)
            board_cache[query] = result
        level = result[21]
        adj_strats = [tuple(result[i:i+3]) for i in range(0, 21, 3)]
        self.print_status(adj_strats, True, level)
        return adj_strats, level

    def get_strat_all(self):
        # tablebase
        if self.board_conn and self.game.GetAgentMode() == 0:
            adj_strats, level = self.query_tablebase()
            if not (adj_strats[0] == (0, 0, 0) or level < 6 or (level < 12 and self.game.GetRealLines() >= 230)):
                if self.game.IsNoAdjMove(*adj_strats[0]):
                    return False, adj_strats[0]
                return True, adj_strats
        # beta
        strat, v = self.get_strat()
        if self.game.IsNoAdjMove(*strat):
            self.print_status([strat]*7, False, [v]*7)
            return False, strat
        return True, self.get_adj_strat(strat)

    def send_seq(self, seq):
        self.request.send(self.gen_seq(seq))

    def step_game(self, strat):
        if self.done: return
        self.game.InputPlacement(*strat)
        self.done = self.game.IsOver()

    def do_premove(self):
        if self.done:
            self.prev = (-1, -1, -1)
            self.send_seq([])
            return
        is_adj, strat = self.get_strat_all()
        if is_adj:
            pos, seq = self.game.GetAdjPremove(strat)
            self.prev = (pos, seq, strat)
            self.step_game(pos)
            self.send_seq(seq)
        else:
            seq = self.game.GetSequence(*strat, True)
            self.prev = (strat, None, None)
            self.send_seq(seq)
            self.send_seq([])

    def finish_move(self, next_piece):
        pos, seq, strats = self.prev
        if self.done:
            if seq is not None: self.send_seq([])
            return
        self.game.SetNextPiece(next_piece)
        if seq is None:
            self.step_game(pos)
            self.prev_placement = pos
        else:
            strat = strats[next_piece]
            fin_seq = self.game.FinishAdjSequence(seq, pos, strat)
            self.send_seq(fin_seq[len(seq):])
            self.step_game(strat)
            self.prev_placement = strat

    def first_piece(self):
        # first piece
        if self.board_conn:
            is_adj, strat = self.get_strat_all()
            if is_adj:
                pos, _ = self.game.GetAdjPremove(strat)
                strat = strat[self.game.GetNextPiece()]
                self.step_game(pos)
            seq = self.game.GetSequence(*strat, not is_adj)
        else:
            strat, _ = self.get_strat()
            seq = self.game.GetSequence(*strat, True)
            if self.game.IsAdjMove(*strat):
                self.step_game(strat)
                strat, _ = self.get_strat()
                seq = self.game.GetSequence(*strat, False)
        self.step_game(strat)
        self.prev_placement = strat
        self.send_seq(seq)
        self.do_premove()

    def handle(self):
        print('Connected')
        self.game = tetris.Tetris()
        while True:
            try:
                data = self.read_until(1)
                if data[0] == 0xff:
                    self.done = False
                    cur, nxt, _ = self.read_until(3)
                    self.game.Reset(cur, nxt)
                    print('New game', (cur, nxt))
                    self.first_piece()
                elif data[0] == 0xfd:
                    r, x, y, nxt = self.read_until(4)
                    if (r, x, y) != self.prev_placement and not self.done:
                        print(f'Error: unexpected placement {(r, x, y)}; expected {self.prev_placement}')
                        self.done = True
                    self.finish_move(nxt)
                    self.do_premove()
            except ConnectionResetError:
                self.request.close()
                break


def GenHandler(models, board_conn):
    class Handler(GameConn):
        def __init__(self, *args, **kwargs):
            self.models = models
            self.board_conn = board_conn
            super().__init__(*args, **kwargs)
    return Handler


def LoadModel(path):
    with torch.no_grad():
        state_dict = torch.load(path)
        channels = state_dict['main_start.0.main.0.weight'].shape[0]
        start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
        end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])
        model = Model(start_blocks, end_blocks, channels).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_normal')
    parser.add_argument('model_single')
    parser.add_argument('model_pushdown')
    parser.add_argument('cache_file')
    parser.add_argument('-b', '--bind', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=3456)
    parser.add_argument('-s', '--server', type=str)
    args = parser.parse_args()
    print(args)

    models = [
        LoadModel(args.model_normal),
        LoadModel(args.model_single),
        LoadModel(args.model_pushdown),
    ]
    try:
        with open(args.cache_file, 'rb') as f:
            board_cache = pickle.load(f)
    except FileNotFoundError:
        pass

    if args.server:
        host, port = args.server.split(':')
        port = int(port)
        board_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        board_conn.connect((host, port))
    else:
        board_conn = None

    # load GPU first to reduce lag
    for model in models:
        model(obs_to_torch(tetris.Tetris().GetState()), pi_only=True)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.bind, args.port), GenHandler(models, board_conn)) as server:
        print(f'Ready, listening on {args.bind}:{args.port}')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
            try:
                with open(args.cache_file, 'rb') as f:
                    board_cache.update(pickle.load(f))
            except FileNotFoundError:
                pass
            with open(args.cache_file, 'wb') as f:
                pickle.dump(board_cache, f)

