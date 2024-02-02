#!/usr/bin/env python3

import argparse, sys, queue, csv, re, random, math
import numpy as np, torch
from multiprocessing import Process, Pipe, Queue, shared_memory

import tetris

from model import Model, obs_to_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 2000
batch_size = 512
n_workers = 2
start_lines = 0
output_file = None
global_seed = 0

class RNG:
    def __init__(self, seed = 0):
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

class Game:
    def __init__(self):
        self.env = tetris.Tetris()
        self.rng = RNG(0)
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

    def reset(self, seed):
        self.seed = seed
        self.rng.reset(seed)
        now = self.rng.spawn()
        nxt = self.rng.spawn()
        self.env.Reset(now, nxt, lines=start_lines)
        self.stats = [0, 0, 0, 0]

def worker_process(remote, q_size, offset, seed_queue, shms):
    games = [Game() for i in range(q_size)]
    is_running = [True for i in range(q_size)]

    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    obs_np = [np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    obs_idx = slice(offset, q_size + offset)

    def Reset(idx):
        nonlocal games, is_running
        try:
            games[idx].reset(seed_queue.get_nowait())
        except queue.Empty:
            is_running[idx] = False

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
            obs = [i.env.GetState() for i in games]
            obs = tuple(zip(*obs))
            for i in range(len(obs)):
                obs_np[i][obs_idx] = np.stack(obs[i])
            remote.send((info, sum(is_running)))
        elif cmd == "close":
            remote.close()
            return

class Worker:
    def __init__(self, *args):
        self.child, parent = Pipe()
        self.process = Process(target=worker_process, args=(parent, *args))
        self.process.start()

@torch.no_grad()
def Main(model):
    assert batch_size % n_workers == 0
    q_size = batch_size // n_workers

    random.seed(global_seed)
    seeds = random.sample(range(2 ** 24), N)
    seed_queue = Queue()
    for i in seeds: seed_queue.put(i)

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

    for i in workers: i.child.send(('init', None))
    for i in workers: i.child.recv()
    started = batch_size
    results = []

    info_arr = []
    old_finished = 0
    while True:
        obs_torch = obs_to_torch(obs_np)
        pi = model(obs_torch, pi_only=True)[0]
        pi = torch.argmax(pi, 1)
        for i in range(n_workers):
            workers[i].child.send(('step', pi[i*q_size:(i+1)*q_size].view(-1).cpu().numpy()))
        to_end = True
        for i in workers:
            info, num_running = i.child.recv()
            info_arr += info
            if num_running > 0: to_end = False
        if to_end: break

        if old_finished // 50 != len(info_arr) // 50:
            text = f'{len(info_arr)} / {N} games finished'
            print(text)
        old_finished = len(info_arr)

    for i in workers:
        i.child.send(('close', None))
        i.child.close()

    if output_file is None:
        writer = csv.writer(sys.stdout)
        writer.writerows(sorted(info_arr))
    else:
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(sorted(info_arr))

    for i in shms:
        i.close()
        i.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-n', '--num', type=int, default=N)
    parser.add_argument('-l', '--start-lines', type=int, default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=batch_size)
    parser.add_argument('-w', '--workers', type=int, default=n_workers)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--seed', type=int, default=global_seed)
    args = parser.parse_args()
    print(args)

    N = args.num
    batch_size = args.batch_size
    n_workers = args.workers
    output_file = args.output
    global_seed = args.seed
    start_lines = args.start_lines

    with torch.no_grad():
        state_dict = torch.load(args.model)
        channels = state_dict['main_start.0.main.0.weight'].shape[0]
        start_blocks = len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)])
        end_blocks = len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)])
        model = Model(start_blocks, end_blocks, channels).to(device)
        model.load_state_dict(state_dict)
        model.eval()

    Main(model)
