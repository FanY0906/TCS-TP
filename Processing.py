import numpy as np


class Processing():
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y


    def __len__(self):
        return len(self.data_X)


    def convert(self, single_seq):
        datas = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '_': 0}
        single_code = []
        for x in single_seq:
            single_code.append(datas[x])
        return np.asarray(single_code)

    def __getitem__(self, idx):
        x = self.data_X[idx]
        x = self.convert(x)
        y = self.data_Y[idx]

        return x, y
