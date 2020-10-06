import torch as th
import numpy as np
import sys

class Device:

    gpu = False
    dummy = None

    @staticmethod
    def set_device(gpuid: int):
        if gpuid != -1:
            Device.gpu = True
            th.cuda.set_device(gpuid)
            sys.stderr.write(f"Device set to GPU {gpuid}\n")
            Device.dummy = Device.float_tensor(1)  # occupy that GPU
        else:
            sys.stderr.write(f"Device set to CPU")

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)
        th.manual_seed(seed)

        if Device.gpu:
            th.cuda.manual_seed(seed)

    @staticmethod
    def float_tensor(*args):
        if Device.gpu:
            return th.cuda.FloatTensor(*args)
        else:
            return th.FloatTensor(*args)

    @staticmethod
    def long_tensor(*args):
        if Device.gpu:
            return th.cuda.LongTensor(*args)
        else:
            return th.LongTensor(*args)

    @staticmethod
    def move(x):
        if Device.gpu:
            return x.cuda()
        else:
            return x

    @staticmethod
    def from_numpy(x):
        if Device.gpu:
            return th.from_numpy(x).cuda()
        else:
            return th.from_numpy(x)

    @staticmethod
    def to_numpy(x):
        if Device.gpu:
            return x.detach().cpu().numpy()
        else:
            return x.detach().numpy()

    @staticmethod
    def move_model(m):
        if Device.gpu:
            m.cuda()

