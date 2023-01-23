import numpy as np
from NonlinearController.lpv_embedding import *
from NonlinearController.model_utils import *

class Model():
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class Model_LPV():
    def __init__(self):
        super(Model_LPV,self).__init__()
        pass

class Model_embedded_LPV():
    def __init__(self, ss_enc):
        super(Model_embedded_LPV,self).__init__()

        self.nx = ss_enc.nx
        self.nu = ss_enc.nu if ss_enc.nu is not None else 1
        self.ny = ss_enc.ny if ss_enc.ny is not None else 1
        
        self.x = np.zeros((self.nx,1))
        self.u = np.zeros((self.nu,1))
        self.y = np.zeros((self.ny,1))

        return