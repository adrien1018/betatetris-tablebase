#!/usr/bin/env python3

import argparse, sys, queue, csv, re, random, math
import numpy as np, torch
from torch.distributions import Categorical
from multiprocessing import Process, Pipe, Queue, shared_memory

import tetris

from model import Model, obs_to_torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 2000
batch_size = 512
n_workers = 2
start_lines = 0
max_lines = None
output_file = None
global_seed = 0
gym_rng = False
clean_only = False
sample_action = False
board_file = None
sample_file = None
start_from_board = False
compile_model = False

class RNGGym:
    def __init__(self, seed=0):
        self.reset(seed)

    def reset(self, seed):
        self.prev = -1
        self.seed = (seed >> 8) & 0xffff
        self.cnt = seed & 0xff
        self.iters = self.cnt >> 4
        if self.iters == 0: self.iters = 16
        self.iters += 3

    def advance(self):
        bit = ((self.seed >> 1) ^ (self.seed >> 9)) & 1
        self.seed = (self.seed >> 1) | (bit << 15)

    def spawn(self):
        for i in range(self.iters): self.advance()
        self.cnt = (self.cnt + 1) & 255
        ind = ((self.seed >> 8) + self.cnt) & 7
        if ind == 7 or ind == self.prev:
            self.advance()
            ind = (((self.seed >> 8) & 7) + [2,7,8,10,11,14,18,0][self.prev]) % 7
        self.prev = ind
        return ind

class RNGNormal:
    def __init__(self, seed=0):
        self.transition_matrix = [
            [1, 5, 6, 5, 5, 5, 5],
            [6, 1, 5, 5, 5, 5, 5],
            [5, 6, 1, 5, 5, 5, 5],
            [5, 5, 5, 2, 5, 5, 5],
            [5, 5, 5, 5, 2, 5, 5],
            [6, 5, 5, 5, 5, 1, 5],
            [5, 5, 5, 5, 6, 5, 1],
        ]
        self.reset(seed)

    def reset(self, seed):
        self.rng = random.Random(seed)
        self.prev = self.rng.randrange(7)

    def spawn(self):
        self.prev = self.rng.choices(range(7), weights=self.transition_matrix[self.prev])[0]
        return self.prev

class Game:
    def __init__(self):
        self.env = tetris.Tetris()
        self.rng = RNGGym(0) if gym_rng else RNGNormal(0)
        self.reset(0)

    def step(self, action):
        LEVELS = [130, 230, 330, 430]
        old_lines = self.env.GetLines()
        old_pieces = self.env.GetPieces()
        r, x, y = action // 200, action // 10 % 20, action % 10
        self.env.InputPlacement(r, x, y)
        lines = self.env.GetLines()
        score = self.env.GetRunScore()
        for i in range(4):
            if old_lines < LEVELS[i] <= lines: self.stats[i] = score
        if self.env.IsOver():
            for i in range(4):
                if LEVELS[i] > lines: self.stats[i] = score
            return False
        if self.env.GetPieces() != old_pieces:
            self.env.SetNextPiece(self.rng.spawn())
        return True

    def get_stats(self):
        return [self.seed] + self.stats + [self.env.GetLines()]

    def reset(self, seed, board=None, now=None):
        self.seed = seed
        self.rng.reset(seed)
        if now is None: now = self.rng.spawn()
        nxt = self.rng.spawn()
        if board:
            self.env.Reset(now, nxt, lines=start_lines, board=board)
        else:
            self.env.Reset(now, nxt, lines=start_lines)
        self.stats = [0, 0, 0, 0]

def worker_process(remote, q_size, offset, seed_queue, shms):
    games = [Game() for i in range(q_size)]
    is_running = [True for i in range(q_size)]

    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    obs_np = [np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    obs_idx = slice(offset, q_size + offset)
    reach_end = False

    def Reset(idx):
        nonlocal games, is_running, reach_end
        if reach_end:
            is_running[idx] = False
            return
        seed, board, piece = seed_queue.get()
        if seed is None:
            reach_end = True
            is_running[idx] = False
            return
        board = tetris.Board(board) if board else None
        if board and board.Count() % 4 != 0:
            board = None
        games[idx].reset(seed, board, piece)
        if games[idx].env.IsOver():
            games[idx].reset(seed)

    while True:
        cmd, data = remote.recv()
        if cmd == 'init':
            for i in range(q_size): Reset(i)
            obs = [i.env.GetState() for i in games]
            obs = tuple(zip(*obs))
            for i in range(len(obs)):
                obs_np[i][obs_idx] = np.stack(obs[i])
            remote.send(0)
        elif cmd == 'step':
            info = []
            for i in range(q_size):
                if not is_running[i]: continue
                if not games[i].step(data[i]):
                    info.append(games[i].get_stats())
                    Reset(i)
                if ((clean_only and not games[i].env.GetBoard().IsCleanForPerfect()) or
                        (max_lines and games[i].env.GetRunLines() >= max_lines)):
                    info.append(games[i].get_stats())
                    Reset(i)

            obs = [i.env.GetState() for i in games]
            obs = tuple(zip(*obs))
            for i in range(len(obs)):
                obs_np[i][obs_idx] = np.stack(obs[i])
            remote.send((info, is_running))
        elif cmd == "close":
            remote.close()
            return

class Worker:
    def __init__(self, *args):
        self.child, parent = Pipe()
        self.process = Process(target=worker_process, args=(parent, *args))
        self.process.start()

@torch.no_grad()
def Main(models):
    assert batch_size % n_workers == 0
    q_size = batch_size // n_workers

    seed_queue = Queue()
    shapes = [(batch_size, *i) for i in tetris.Tetris.StateShapes()]
    types = [np.dtype(i) for i in tetris.Tetris.StateTypes()]
    shms = [
        shared_memory.SharedMemory(create=True, size=math.prod(shape) * typ.itemsize)
        for shape, typ in zip(shapes, types)
    ]
    obs_np = [
        np.ndarray(shape, dtype=typ, buffer=shm.buf)
        for shm, shape, typ in zip(shms, shapes, types)
    ]
    shm_child = [(shm.name, shape, typ) for shm, shape, typ in zip(shms, shapes, types)]
    workers = [Worker(q_size, i * q_size, seed_queue, shm_child) for i in range(n_workers)]

    if board_file:
        board_set = set()
        try:
            with open(board_file, 'rb', buffering=1048576) as f:
                while True:
                    item = f.read(25)
                    if len(item) == 0: break
                    assert len(item) == 25
                    board_set.add(item)
        except FileNotFoundError:
            pass
        last_save = time.time()
        save_num = 0

    random.seed(global_seed)
    seeds = random.sample(range(2 ** 24 if gym_rng else 2 ** 60), N)
    if sample_file:
        boards = []
        with open(sample_file, 'rb', buffering=1048576) as f:
            while True:
                item = f.read(26)
                if len(item) == 0: break
                assert len(item) == 26
                boards.append((item[:25], item[25] & 7))
        boards = boards * (N // len(boards)) + random.sample(boards, N % len(boards))
    elif board_file and start_from_board and len(board_set) > 0:
        pn = int(N * 0.8)
        boards = random.choices(list(board_set), k=pn) + [None] * (N - pn)
        random.shuffle(boards)
        boards = [(i, None) for i in boards]
    else:
        boards = [(None, None)] * N
    for i in zip(seeds, boards): seed_queue.put((i[0], *i[1]))
    for i in range(n_workers * 2): seed_queue.put((None, None, None))

    def Save(fname, to_sort=False):
        nonlocal last_save
        with open(fname, 'wb', buffering=1048576) as f:
            if to_sort:
                for i in sorted(board_set): f.write(i)
            else:
                for i in board_set: f.write(i)
        last_save = time.time()

    for i in workers: i.child.send(('init', None))
    for i in workers: i.child.recv()
    started = batch_size
    results = []

    try:
        info_arr = []
        old_finished = 0
        running = np.ones(batch_size, dtype='bool')
        while True:
            obs_torch = obs_to_torch([i[running] for i in obs_np])

            if board_file:
                boards = obs_torch[0][:,0].view(-1, 200).cpu().numpy().astype('bool')
                boards = np.packbits(boards, axis=1, bitorder='little')
                board_set.update(map(lambda x: x.tobytes(), boards))

            pi = torch.stack([model(obs_torch, pi_only=True)[0] for model in models]).mean(0)
            if sample_action:
                pi = Categorical(logits=pi*0.75).sample()
            else:
                pi = torch.argmax(pi, 1)
            pi_np = np.zeros(batch_size, dtype='int')
            pi_np[running] = pi.view(-1).cpu().numpy()
            for i in range(n_workers):
                workers[i].child.send(('step', pi_np[i*q_size:(i+1)*q_size]))
            to_end = True
            for i in range(n_workers):
                info, is_running = workers[i].child.recv()
                info_arr += info
                running[i*q_size:(i+1)*q_size] = is_running
            if running.sum() == 0: break

            if old_finished // 50 != len(info_arr) // 50:
                text = f'{len(info_arr)} / {N} games finished'
                if board_file: text += f'; {len(board_set)} boards collected'
                print(text, file=sys.stderr)
            old_finished = len(info_arr)

            if board_file and time.time() - last_save >= 5400:
                Save(board_file + f'.{save_num}')
                save_num = 1 - save_num
    finally:
        if board_file: Save(board_file)

        if output_file is None:
            writer = csv.writer(sys.stdout)
            writer.writerows(sorted(info_arr))
        else:
            with open(output_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(sorted(info_arr))

        for i in workers:
            i.child.send(('close', None))
            i.child.close()
        for i in shms:
            i.close()
            i.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    parser.add_argument('-n', '--num', type=int, default=N)
    parser.add_argument('-l', '--start-lines', type=int, default=0)
    parser.add_argument('-m', '--max-lines', type=int)
    parser.add_argument('-b', '--batch-size', type=int, default=batch_size)
    parser.add_argument('-w', '--workers', type=int, default=n_workers)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--seed', type=int, default=global_seed)
    parser.add_argument('--gym-rng', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--sample-action', action='store_true')
    parser.add_argument('--board-file', type = str)
    parser.add_argument('--sample-file', type = str)
    parser.add_argument('--start-from-board', action='store_true')
    parser.add_argument('--compile-model', action='store_true')
    args = parser.parse_args()
    print(args, file=sys.stderr)

    N = args.num
    batch_size = args.batch_size
    n_workers = args.workers
    output_file = args.output
    global_seed = args.seed
    start_lines = args.start_lines
    max_lines = args.max_lines
    gym_rng = args.gym_rng
    clean_only = args.clean_only
    sample_action = args.sample_action
    board_file = args.board_file
    sample_file = args.sample_file
    start_from_board = args.start_from_board
    compile_model = args.compile_model

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    models = []
    for model_file in args.models:
        with torch.no_grad():
            state_dict = torch.load(model_file)
            channels = state_dict['main_start.0.main.0.weight'].shape[0]
            start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
            end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])
            model = Model(start_blocks, end_blocks, channels).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            if compile_model:
                model = torch.compile(model)
            models.append(model)

    Main(models)
