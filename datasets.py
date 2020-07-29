from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DeltasDataset(Dataset):
    def __init__(self, csv_file):
        self.deltas_frame = pd.read_csv(csv_file)

    def __len__(self):
        return  len(self.deltas_frame)

    def __getitem__(self, idx):
        delta = self.deltas_frame.iloc[idx,:]


        delta = np.array([delta])
        delta=(delta-delta.mean())/delta.std()
        return delta

