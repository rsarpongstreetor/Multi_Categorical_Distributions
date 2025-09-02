import torch
from torch.distributions import Categorical, Distribution
from typing import List


class MultiCategorical(Distribution):

    def __init__(self, dists: List[Categorical]):
        super().__init__()
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)


def multi_categorical_maker(nvec):
    def get_multi_categorical(logits):
        start = 0
        ans = []
        for n in nvec:
            ans.append(Categorical(logits=logits[:, start: start + n]))
            start += n
        return MultiCategorical(ans)
    return get_multi_categorical
