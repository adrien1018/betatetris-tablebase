import hashlib, traceback
import numpy as np
import torch
from multiprocessing import shared_memory
from torch.multiprocessing import Process, Pipe

import tetris

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(seed)
        self.reset()

    def step(self, action):
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
            is_over[0] = True
            self.reset()
        return self.env.GetState(), reward, is_over, info

    def reset(self):
        self.reward = 0.
        self.env.ResetRandom()
        return self.env.GetState()

def worker_process(remote, name: str, shms: list, idx: slice, seed: int):
    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    shms_np = [np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    shm_obs = tuple(shms_np[:-2])
    shm_reward, shm_over = tuple(shms_np[-2:])

    # create game environments
    num = idx.stop - idx.start
    Seed = lambda x: int.from_bytes(hashlib.sha256(
        int.to_bytes(seed, 8, 'little') + int.to_bytes(x, 4, 'little')).digest(), 'little')
    games = [Game(Seed(i)) for i in range(num)]
    # wait for instructions from the connection and execute them
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                step, actions, epoch = data
                result = []
                for i in range(num):
                    result.append(games[i].step(actions[i]))
                obs, reward, over, info = zip(*result)
                obs = tuple(zip(*obs))
                for i in range(len(obs)):
                    shm_obs[i][idx] = np.stack(obs[i])
                shm_reward[idx,step] = np.stack(reward)
                shm_over[idx,step] = np.stack(over)
                info = list(filter(lambda x: x is not None, info))
                remote.send(info)
            elif cmd == "reset":
                obs = [games[i].reset() for i in range(num)]
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
