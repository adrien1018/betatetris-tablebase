import os, argparse
from typing import Optional

from labml import experiment
from labml.configs import BaseConfigs, FloatDynamicHyperParam

class Configs(BaseConfigs):
    # #### Configurations
    ## NN
    start_blocks: int = 1
    end_blocks: int = 4
    channels: int = 192

    def model_args(self):
        return (self.start_blocks, self.end_blocks, self.channels)

    ## training
    lr: float = FloatDynamicHyperParam(1e-4, range_ = (0, 1e-3))
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = FloatDynamicHyperParam(0.992 ** 0.5, range_ = (0.95, 1))
    lamda: float = FloatDynamicHyperParam(0.93, range_ = (0.8, 1))
    # number of updates
    updates: int = 400000
    # number of epochs to train the model with sampled data
    epochs: int = 1
    # number of worker processes
    n_workers: int = 2
    env_per_worker: int = 48
    # number of steps to run on each process for a single update
    worker_steps: int = 256
    # size of mini batches
    n_update_per_epoch: int = 32
    # calculate loss in batches of mini_batch_size
    mini_batch_size: int = 768

    ## loss calculation
    clipping_range: float = 0.2
    vf_weight: float = FloatDynamicHyperParam(0.5, range_ = (0, 1))
    entropy_weight: float = FloatDynamicHyperParam(2e-2, range_ = (0, 5e-2))
    reg_l2: float = FloatDynamicHyperParam(0., range_ = (0, 5e-5))

    save_interval: int = 500


def LoadConfig(with_experiment = True):
    parser = argparse.ArgumentParser()
    if with_experiment:
        parser.add_argument('name')
        parser.add_argument('uuid', nargs = '?', default = '')
        parser.add_argument('checkpoint', nargs = '?', type = int, default = None)
        parser.add_argument('--ignore-optimizer', action = 'store_true')
    conf = Configs()
    keys = conf._to_json()
    dynamic_keys = set()
    for key in keys:
        ptype = type(conf.__getattribute__(key))
        if ptype == FloatDynamicHyperParam:
            ptype = float
            dynamic_keys.add(key)
        parser.add_argument('--' + key.replace('_', '-'), type = ptype)

    args, others = parser.parse_known_args()
    args = vars(args)
    override_dict = {}
    for key in keys:
        if key not in dynamic_keys and args[key] is not None: override_dict[key] = args[key]
    conf = Configs()
    for key in dynamic_keys:
        if args[key] is not None:
            conf.__getattribute__(key).set_value(args[key])
    if with_experiment:
        os.makedirs('logs/{}'.format(args['name']), exist_ok = True)
        uuid = 1
        while os.path.exists('logs/{0}/{0}-{1:03d}'.format(args['name'], uuid)):
            uuid += 1
        experiment.create(name = args['name'], uuid = '{0}-{1:03d}'.format(args['name'], uuid))
        experiment.configs(conf, override_dict)
    else:
        for key, val in override_dict:
            conf.__setattr__(key, val)
    return conf, args, others
