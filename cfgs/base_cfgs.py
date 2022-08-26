import os
import torch
import random
import numpy as np

from cfgs.path_cfgs import PATH
from types import MethodType


class ExpConfig(PATH):
    """
    Configuration object for model and experiments.
    """
    def __init__(self):
        super(ExpConfig, self).__init__()

        # Set Devices
        # If use multi-gpu training, set e.g.'0, 1, 2' instead
        self.GPU = '0'

        # Set RNG For CPU And GPUs
        self.SEED = random.randint(0, 99999999)

        # Define a random seed for new training
        self.VERSION = 'default'

        # For resuming training and testing
        self.CKPT_VERSION = self.VERSION + '_' + str(self.SEED)
        self.CKPT_EPOCH = 0

        # Absolute checkpoint path, override 'CKPT_VERSION' and 'CKPT_EPOCH
        self.CKPT_PATH = None

        # Set training split
        self.TRAIN_SPLIT = 'train'

        # Define data split
        self.SPLIT = {
            'train': '', 'valid': 'valid', 'test': 'test'
        }

        # Optimizer
        self.OPT = ''
        self.OPT_PARAMS = {}

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def setup(self):
        def all(iterable):
            for element in iterable:
                if not element:
                    return False
            return True

        assert self.RUN_MODE in ['train', 'val', 'test'], "Please select a mode"

        # ---------- Setup devices ----------
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)

        # ---------- Setup seed ----------
        # set pytorch seed
        torch.manual_seed(self.SEED)
        if self.N_GPU < 2:
            torch.cuda.manual_seed(self.SEED)
        else:
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True

        # set numpy and random seed, in case it is needed
        np.random.seed(self.SEED)
        random.seed(self.SEED)

        # ---------- Setup Opt ----------
        assert self.OPT in ['Adam', 'AdamW', 'RMSProp', 'SGD', 'Adagrad']
        optim = getattr(torch.optim, self.OPT)
        default_params_dict = dict(zip(optim.__init__.__code__.co_varnames[3: optim.__init__.__code__.co_argcount],
                                       optim.__init__.__defaults__[1:]))

        assert all(list(map(lambda x: x in default_params_dict, self.OPT_PARAMS)))

        for key in self.OPT_PARAMS:
            if isinstance(self.OPT_PARAMS[key], str):
                self.OPT_PARAMS[key] = eval(self.OPT_PARAMS[key])
            else:
                print("To avoid ambiguity, set the value of 'OPT_PARAMS' to string type")
                exit(-1)
        self.OPT_PARAMS = {**default_params_dict, **self.OPT_PARAMS}

        if self.CKPT_PATH is not None:
            print("CKPT_VERSION will not work with CKPT_PATH")
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(random.randint(0, 99999999))

        if self.CKPT_VERSION.split('_')[0] != self.VERSION and self.RUN_MODE in ['val', 'test']:
            self.VERSION = self.CKPT_VERSION

        # ---------- Setup split ----------
        self.SPLIT['train'] = self.TRAIN_SPLIT

    def config_dict(self):
        conf_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and not isinstance(getattr(self, attr), MethodType):
                conf_dict[attr] = getattr(self, attr)
        return conf_dict

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('_') and not isinstance(getattr(self, attr), MethodType):
                print('{{{: <17}}}->'.format(attr), getattr(self, attr))

        return ''
