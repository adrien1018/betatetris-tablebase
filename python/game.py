import hashlib, traceback, os.path, random
from typing import Optional
from multiprocessing import shared_memory
import numpy as np
import torch
from torch.multiprocessing import Process, Pipe
from filelock import FileLock

import tetris

def SpeedFromLines(x):
    if x < 130: return 0
    if x < 230: return 1
    if x < 330: return 2
    return 3

def RandLinesFromSpeed(x):
    LINE_TABLE = [0, 130, 230, 330, 430]
    return random.randrange(LINE_TABLE[x]//2, LINE_TABLE[x+1]//2)*2

class BoardManager:
    def __init__(self, board_file):
        self.eof = False
        self.board_file = board_file
        self.normal_cnt = 0
        self.board_cnt = 0
        self._NextBatch()

    def AddCnt(self, is_normal: bool, pieces: int):
        if is_normal:
            self.normal_cnt += pieces
        else:
            self.board_cnt += pieces

    def GetNewBoard(self):
        if not self.eof and self.normal_cnt * 1.5 > self.board_cnt:
            if self.data_offset >= len(self.data):
                self._NextBatch()
                if self.data_offset >= len(self.data):
                    return None
            b, piece, level = self.data[self.data_offset]
            cells = 200 - int.from_bytes(b, 'little').bit_count()
            lines = RandLinesFromSpeed(level)
            if cells % 4 != 0: lines += 1
            self.data_offset += 1
            return tetris.Board(b), piece, lines
        return None

    def _NextBatch(self):
        if not self.board_file:
            self.eof = True
            return
        self.data = self.read_board_file(self.board_file)
        self.data_offset = 0

    @staticmethod
    def read_board_file(board_file: str, chunk_size: int = 4096):
        DATA_SIZE = 26
        board_file_offset_f = board_file + '.offset'
        board_file_lock = board_file + '.lock'
        lock = FileLock(board_file_lock)
        with lock:
            if os.path.isfile(board_file_offset_f):
                with open(board_file_offset_f, 'r') as f: offset = int(f.read().strip())
            else:
                offset = 0
            with open(board_file, 'rb') as f:
                f.seek(offset * DATA_SIZE)
                data = f.read(chunk_size * DATA_SIZE)
                ret = [(data[i*DATA_SIZE:i*DATA_SIZE+DATA_SIZE-1],
                        data[i*DATA_SIZE+DATA_SIZE-1]&7,
                        data[i*DATA_SIZE+DATA_SIZE-1]>>3) for i in range(len(data) // DATA_SIZE)]
            with open(board_file_offset_f, 'w') as f:
                print(0 if len(ret) == 0 else offset + len(ret), file=f)
        return ret

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(seed)
        self.is_normal = True
        self.start_speed = 0
        self.speed_cnt = np.array([0, 0, 0, 0])
        self.reset()

    def step(self, action, manager=None):
        r, x, y = action // 200, action // 10 % 20, action % 10
        reward = np.array(self.env.InputPlacement(r, x, y))
        self.reward += reward[0]

        info = None
        is_over = np.array([False, False])
        if self.env.IsOver():
            info = {'reward': self.reward,
                    'score': self.env.GetRunScore(),
                    'lines': self.env.GetRunLines(),
                    'pieces': self.env.GetRunPieces()}
            if manager: manager.AddCnt(self.is_normal, info['pieces'])
            self.speed_cnt[self.start_speed] += info['pieces']
            is_over[0] = True
            self.reset(manager)
        return self.env.GetState(), reward, is_over, info

    def reset(self, manager=None):
        self.reward = 0.
        self.start_speed = np.argmin(self.speed_cnt / np.array([2., 1.6, 1.3, 1.]))
        blank_lines = RandLinesFromSpeed(self.start_speed)

        if manager:
            data = manager.GetNewBoard()
            if data is None:
                self.is_normal = True
                self.env.ResetRandom(lines=blank_lines)
            else:
                self.is_normal = False
                while True:
                    self.env.Reset(data[1], random.randrange(7), lines=data[2], board=data[0])
                    self.start_speed = SpeedFromLines(data[2])
                    if not self.env.IsOver(): break
                    data = manager.GetNewBoard()
                    if not data:
                        self.is_normal = True
                        self.env.ResetRandom(lines=blank_lines)
                        break
        else:
            self.is_normal = True
            self.env.ResetRandom(lines=blank_lines)
        return self.env.GetState()

def worker_process(remote, name: str, shms: list, idx: slice, seed: int, board_file: Optional[str]):
    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    shms_np = [np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    shm_obs = tuple(shms_np[:-2])
    shm_reward, shm_over = tuple(shms_np[-2:])

    # create game environments
    num = idx.stop - idx.start
    Seed = lambda x: int.from_bytes(hashlib.sha256(
        int.to_bytes(seed, 8, 'little') + int.to_bytes(x, 8, 'little')).digest(), 'little')
    random.seed(Seed(12345))
    games = [Game(Seed(i)) for i in range(num)]
    manager = BoardManager(board_file)
    # wait for instructions from the connection and execute them
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                step, actions, epoch = data
                result = []
                for i in range(num):
                    result.append(games[i].step(actions[i], manager))
                obs, reward, over, info = zip(*result)
                obs = tuple(zip(*obs))
                for i in range(len(obs)):
                    shm_obs[i][idx] = np.stack(obs[i])
                shm_reward[idx,step] = np.stack(reward)
                shm_over[idx,step] = np.stack(over)
                info = list(filter(lambda x: x is not None, info))
                remote.send(info)
            elif cmd == "reset":
                obs = [games[i].reset(manager) for i in range(num)]
                obs = tuple(zip(*obs))
                for i in range(len(obs)):
                    shm_obs[i][idx] = np.stack(obs[i])
                remote.send(0)
            elif cmd == "close":
                # remote will be closed on finally
                return
            elif cmd == "set_param":
                _ = data
                for i in games:
                    # i.pre_trans = pre_trans
                    pass
            else:
                raise NotImplementedError
    except:
        print(traceback.format_exc())
        raise
    finally:
        remote.close()
        for i in shms: i[0].close()

class Worker:
    """Creates a new worker and runs it in a separate process."""
    def __init__(self, *args):
        self.child, parent = Pipe()
        self.process = Process(target=worker_process, args=(parent, *args))
        self.process.start()
