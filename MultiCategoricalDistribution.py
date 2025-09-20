# Here is the current query...import torch
import torch
from torch.distributions import Categorical, Distribution
from typing import List
import torch.nn as nn
import torch.nn as nn


class MultiCategorical(Distribution):

    def __init__(self, dists: List[Categorical]):
        super().__init__()
        if not all(isinstance(d, Categorical) for d in dists):
             raise TypeError("All distributions in the list must be Categorical.")
        if not dists:
             raise ValueError("Distribution list cannot be empty.")
        self.dists = list(dists) # Use a standard list

    def log_prob(self, value):
        if value.shape[-1] != len(self.dists):
             raise ValueError(f"Value shape {value.shape} last dimension must match number of distributions {len(self.dists)}")
        ans = []
        # Ensure splitting happens on the last dimension where individual actions are stacked
        # Value shape is expected to be (batch_size, num_agents, num_actions_per_feature)
        # Split along the last dimension (num_actions_per_feature)
        values_split = torch.split(value, 1, dim=-1)
        if len(values_split) != len(self.dists):
             raise ValueError(f"Splitting value tensor resulted in {len(values_split)} tensors, but expected {len(self.dists)}")

        for d, v in zip(self.dists, values_split):
            # Squeeze the last dimension of v, which is 1 after splitting
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1) # Sum log_probs across the action components


    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1) # Sum entropies


    def sample(self, sample_shape=torch.Size()):
        # Sample from each categorical distribution and stack the results
        # Expected output shape: (*batch_shape, num_actions_per_feature)
        samples = [d.sample(sample_shape) for d in self.dists]
        return torch.stack(samples, dim=-1)

# Inside your MultiCategorical class definition in MultiCategoricalDistribution.py

    @property
    def deterministic_sample(self) -> torch.Tensor:
        """
        Returns the mode of the MultiCategorical distribution.
        This corresponds to the action with the highest probability for each
        of the individual categorical choices.
        """
        # Assuming self.logits has shape [..., num_agents, num_individual_actions_features * num_action_categories]
        # And the logits are grouped by action feature, then category
        # Reshape logits to [..., num_agents, num_individual_actions_features, num_action_categories]
        logits_reshaped = self.logits.view(
            *self.logits.shape[:-1],
            self.action_spec.shape[-2], # num_individual_actions_features
            self.action_spec.space.nvec[0].item() # num_action_categories (assuming all have the same n)
        )
    
        # Get the index of the maximum logit for each individual action feature
        # The action is the index with the highest probability
        mode = torch.argmax(logits_reshaped, dim=-1) # Shape [..., num_agents, num_individual_actions_features]
    
        # The action spec is MultiCategorical with shape [num_agents, num_individual_actions_features]
        # The sampled action should match this shape.
        return mode.to(self.action_spec.dtype) # Ensure dtype matches action spec
    
    
    # You might also need a .mode property for compatibility with some torchrl components
    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution. Alias for deterministic_sample.
        """
        return self.deterministic_sample





def multi_categorical_maker(nvec):
    # nvec: List of number of categories for each individual action component
    # Example: [3, 3, 3, ..., 3] (13 times)
    def get_multi_categorical(logits):
        # logits shape is expected to be (batch_size, num_agents, sum(nvec))
        if logits.shape[-1] != sum(nvec):
             raise ValueError(f"Logits shape {logits.shape} last dimension must match sum of nvec {sum(nvec)}")

        dists = []
        start = 0
        for n in nvec:
            # Correctly slice along the last dimension (the flattened categories)
            # Logits slice shape will be (batch_size, num_agents, n)
            dists.append(Categorical(logits=logits[..., start : start + n]))
            start += n

        return MultiCategorical(dists)
    return get_multi_categorical
