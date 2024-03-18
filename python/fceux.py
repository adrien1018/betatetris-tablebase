#!/usr/bin/env python3

import sys, os.path, socketserver, argparse, re, socket
import numpy as np, torch
import curses

import tetris
from model import Model, obs_to_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stdscr = None
args = None
thresholds = None


def myprint(*pargs, posx=0, posy=0, clear=True, refresh=True):
    if args.use_curses:
        if clear: stdscr.clear()
        pstr = ' '.join(map(str, pargs))
        stdscr.addstr(posx, posy, pstr)
        if refresh: stdscr.refresh()
    else:
        print(*pargs, flush=True)


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
        if len(seq) == 0: return bytes([0xfe, 1, 0])
        return bytes([0xfe, len(seq)]) + seq.tobytes()

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
        with torch.no_grad():
            pi, v = self.model(obs_to_torch(self.game.GetState()), pi_only=True)
            action = torch.argmax(pi, 1).item()
            return (action // 200, action // 10 % 20, action % 10), v[0].item()

    def print_status(self, strats, is_table, data, refresh=True):
        def StratText(i, cur_piece):
            PIECE_TABLE = 'TJZOSLI'
            ROT_TABLE = ['dlur', 'dlur', ['',''], [''], ['', ''], 'dlur', ['', '']]
            COL_TABLE = [[(-1,2),(-1,1),(-1,2),(0,2)],
                         [(-1,2),(-1,1),(-1,2),(0,2)],
                         [(-1,2),(0,2)],
                         [(-1,1)],
                         [(-1,2),(0,2)],
                         [(-1,2),(-1,1),(-1,2),(0,2)],
                         [(-2,2),(0,1)]]
            r,x,y = strats[i]
            ori = PIECE_TABLE[cur_piece] + ROT_TABLE[cur_piece][r] + '-'
            for j in range(*COL_TABLE[cur_piece][r]):
                ori += str(j + y + 1)[-1]
            ntxt = f'{ori:6s}@row {20-x:2d}'
            if not is_table:
                ntxt += f'; val {data[i]:6.3f}'
            return ntxt

        txt = 'Game #' + str(self.games) + '\n'
        txt += 'Games to 19: ' + str(self.num_19) + '\n'
        txt += 'Games to 29: ' + str(self.num_29) + '\n\nAgent: '
        txt += 'Tablebase' if is_table else 'Neural Net'
        txt += '\nPlacements: (where to place\n the upcoming piece based on\n its next piece)\n'
        for i in range(7):
            txt += 'TJZOSLI'[i] + ': ' + StratText(i, self.game.GetNowPiece()) + '\n'
        if is_table:
            txt += f'confidence {data}'
            if thresholds:
                multiplier = (args.buckets - 2) / (args.ratio_high - args.ratio_low)
                bias = 1 - args.ratio_low * multiplier
                value_low = 0.0 if data == 0 else (data - bias) / multiplier
                value_high = 1.0 if data == args.buckets - 1 else (data + 1 - bias) / multiplier
                scale = thresholds[self.game.GetLines()] * 100
                value_low *= scale
                value_high *= scale
                txt += f' (pure tablebase\n probability to 29:\n{value_low:6.2f}% -{value_high:6.2f}%)'
        txt += '\n'
        myprint(txt, refresh=refresh)

    def get_adj_strat(self, pos):
        with torch.no_grad():
            pi, v = self.model(obs_to_torch(self.game.GetAdjStates(*pos)), pi_only=True)
            actions = torch.argmax(pi, 1).flatten().cpu().tolist()
            strats = [(action // 200, action // 10 % 20, action % 10) for action in actions]
            self.print_status(strats, False, v[0].flatten().cpu().tolist())
            return strats

    def query_tablebase(self):
        query = (
            bytes([1]) +
            self.game.GetBoard().GetBytes() +
            bytes([self.game.GetNowPiece(), self.game.GetLines() % 256, self.game.GetLines() // 256])
        )
        self.board_conn.sendall(query)
        result = self.board_conn.recv(22)
        level = result[21]
        adj_strats = [tuple(result[i:i+3]) for i in range(0, 21, 3)]
        self.print_status(adj_strats, True, level, refresh=False)
        return adj_strats, level

    def get_strat_all(self):
        if self.game.GetLines() < 230 and self.board_conn:
            # tablebase
            adj_strats, level = self.query_tablebase()
            if not adj_strats[0] == (0, 0, 0):
                stdscr.refresh()
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
            seq = self.game.GetSequence(*strat)
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
            else:
                strat = adj_strats
            seq = self.game.GetSequence(*strat)
        else:
            strat, _ = self.get_strat()
            seq = self.game.GetSequence(*strat)
            if self.game.IsAdjMove(*strat):
                self.step_game(strat)
                strat, _ = self.get_strat()
                seq = self.game.GetSequence(*strat)
        self.step_game(strat)
        self.prev_placement = strat
        self.send_seq(seq)
        self.do_premove()

    def handle(self):
        myprint('Connected')
        self.game = tetris.Tetris()
        while True:
            try:
                data = self.read_until(1)
                if data[0] == 0xff:
                    self.done = False
                    cur, nxt, _ = self.read_until(3)
                    if self.games != 0:
                        self.num_19 += int(self.game.GetLines() >= 130)
                        self.num_29 += int(self.game.GetLines() >= 230)
                    self.games += 1
                    self.game.Reset(cur, nxt)
                    myprint('New game', (cur, nxt))
                    self.first_piece()
                elif data[0] == 0xfd:
                    r, x, y, nxt = self.read_until(4)
                    if (r, x, y) != self.prev_placement and not self.done:
                        myprint(f'Error: unexpected placement {(r, x, y)}; expected {self.prev_placement}')
                        self.done = True
                    self.finish_move(nxt)
                    self.do_premove()
            except ConnectionResetError:
                self.request.close()
                break


def GenHandler(model, board_conn):
    class Handler(GameConn):
        def __init__(self, *args, **kwargs):
            self.model = model
            self.board_conn = board_conn
            self.games = 0
            self.num_19 = 0
            self.num_29 = 0
            super().__init__(*args, **kwargs)
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-b', '--bind', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=3456)
    parser.add_argument('-s', '--server', type=str)
    parser.add_argument('-c', '--use-curses', action='store_true')
    parser.add_argument('--threshold-file', type=str)
    parser.add_argument('--ratio-low', type=float, default=0.01)
    parser.add_argument('--ratio-high', type=float, default=1.0)
    parser.add_argument('--buckets', type=float, default=256)
    args = parser.parse_args()
    print(args)

    if args.threshold_file:
        with open(args.threshold_file, 'r') as f:
            thresholds = list(map(float, f.read().split()))

    with torch.no_grad():
        state_dict = torch.load(args.model)
        channels = state_dict['main_start.0.main.0.weight'].shape[0]
        start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
        end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])
        model = Model(start_blocks, end_blocks, channels).to(device)
        model.load_state_dict(state_dict)
        model.eval()

    if args.server:
        host, port = args.server.split(':')
        port = int(port)
        board_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        board_conn.connect((host, port))
    else:
        board_conn = None

    # load GPU first to reduce lag
    model(obs_to_torch(tetris.Tetris().GetState()), pi_only=True)

    if args.use_curses:
        stdscr = curses.initscr()
        curses.noecho()
        curses.curs_set(False)

    try:
        with socketserver.TCPServer((args.bind, args.port), GenHandler(model, board_conn)) as server:
            server.model = model
            myprint(f'Ready, listening on {args.bind}:{args.port}')
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.shutdown()
                if args.use_curses:
                    curses.curs_set(True)
                    curses.echo()
                    curses.endwin()
    except:
        if args.use_curses:
            curses.echo()
            curses.endwin()
        raise

