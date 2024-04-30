import torch
import numpy as np

# Phoneme spectra dataset
# (Num examples, num channels, num time steps,)
# Shape of train dataset: (3315, 11, 217)
# Shape of valid dataset: (1677, 11, 217)
# Shape of test dataset: (1676, 11, 217)

TRAIN_MAX = 126.76
TRAIN_MIN = 0.0

CLASS_MAPPING = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5,
                 'B': 6, 'CH': 7, 'D': 8, 'DH': 9, 'EH': 10, 'ER': 11,
                 'EY': 12, 'F': 13, 'G': 14, 'HH': 15, 'IH': 16, 'IY': 17,
                 'JH': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'NG': 23,
                 'OW': 24, 'OY': 25, 'P': 26, 'R': 27, 'S': 28, 'SH': 29,
                 'T': 30, 'TH': 31, 'UH': 32, 'UW': 33, 'V': 34, 'W': 35, 
                 'Y': 36, 'Z': 37, 'ZH': 38}

def minmaxscaler(data):
    return (data - TRAIN_MIN) / (TRAIN_MAX - TRAIN_MIN)

class PhonemeDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, label_filename, transform=None):
        self.data = np.load(data_filename)
        self.target = torch.from_numpy(np.array([CLASS_MAPPING[label] for label in np.load(label_filename)]))

        # Pad the data to 220 timesteps
        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, 3)), mode='constant', constant_values=0)

        # Scale the data
        self.data = minmaxscaler(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]