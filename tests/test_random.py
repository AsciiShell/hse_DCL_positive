import torch
import numpy as np
import scipy.stats


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, max_val):
        super().__init__()
        self.a = np.arange(max_val)

    def __len__(self):
        return len(self.a) * 100

    def __getitem__(self, index):
        return np.random.choice(self.a)


def test_simple():
    assert 1 == int("1")


def test_dataloader():
    MAX_VAL = 10000
    loader = torch.utils.data.DataLoader(
        RandomDataset(MAX_VAL),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    a = []
    for b in loader:
        a.extend(b.detach().cpu().numpy())
    rnd = scipy.stats.uniform(loc=0.0, scale=MAX_VAL)
    pvalue = scipy.stats.ks_2samp(np.array(a),  rnd.rvs(len(a))).pvalue
    assert pvalue > 0.01
