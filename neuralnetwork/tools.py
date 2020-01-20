import numpy as np
class Tools():
    def unroll(self, a, b):
        a = np.vstack(a.reshape(a.size,order='F'))
        b = np.vstack(b.reshape(b.size,order='F'))
        return np.concatenate((a, b), axis=0)

    