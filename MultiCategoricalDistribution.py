import torch
from torch.distributions import Categorical, Distribution
from typing import List

# Assuming MultiCategorical class is defined elsewhere in this file and imported
# from .MultiCategoricalDistribution import MultiCategorical # If in the same package

# If MultiCategorical is defined within this file:
class MultiCategorical(Distribution):

    def __init__(self, dists: List[Categorical]):
        super().__init__()
        if not all(isinstance(d, Categorical) for d in dists):
             raise TypeError("All distributions in the list must be Categorical.")
        if not dists:
             raise ValueError("Distribution list cannot be empty.")
        self.dists = list(dists) # Use a standard list
        # Determine the total number of categories by summing categories of individual distributions
        self.total_categories = sum(d.param_shape[-1] for d in self.dists)
        # Determine the event shape based on the structure of individual distributions
        # Assuming each Categorical represents one sub-action feature for an agent
        # The total event shape is the concatenation of event shapes of individual distributions
        # If each Categorical has an event_shape of [], the combined event shape is [len(dists)]
        # If the MultiCategorical represents actions for multiple agents and multiple sub-actions per agent,
        # the event shape might be [num_agents, num_sub_actions_per_agent].
        # Let's assume for now the MultiCategorical wraps distributions for a flattened action space
        # of size num_agents * num_sub_distributions. The event shape would then be [len(dists)].
        # If the distribution is intended to output a structured action [num_agents, num_sub_distributions],
        # the event shape needs to reflect that.
        # Based on the environment action_spec: shape=torch.Size([5, 4, 13]), this suggests the action for a single env
        # is shaped [4, 13]. So the event shape of the *unbatched* distribution should be [4, 13].
        # The number of individual distributions (len(dists)) should be num_agents * num_sub_distributions_per_agent = 4 * 13 = 52.
        # Each distribution in self.dists corresponds to one of these 52 individual categorical choices.
        # The event shape should represent the shape of a *single* sample from this distribution.
        # If sampling returns [num_agents, num_sub_distributions], the event shape is [num_agents, num_sub_distributions].

        # Let's infer the event shape from the individual distributions' event shapes and the expected total structure.
        # Assuming self.dists is a flattened list of Categorical distributions covering all agents and sub-actions.
        # The total number of individual categorical choices is len(self.dists).
        # If the final action needs to be shaped [num_agents, num_sub_distributions_per_agent],
        # and len(self.dists) = num_agents * num_sub_distributions_per_agent,
        # the event shape should be [num_agents, num_sub_distributions_per_agent].
        # Let's assume num_agents = 4 and num_sub_distributions_per_agent = 13 based on the environment spec.
        self.num_agents = 4 # Hardcoded based on env setup
        self.num_sub_distributions_per_agent = 13 # Hardcoded based on env setup
        # Verify that the number of distributions matches the expected flattened size
        if len(self.dists) != self.num_agents * self.num_sub_distributions_per_agent:
             raise ValueError(f"Number of individual distributions ({len(self.dists)}) does not match expected number of categorical choices ({self.num_agents} * {self.num_sub_distributions_per_agent}).")

        self._event_shape = torch.Size([self.num_agents, self.num_sub_distributions_per_agent]) # Set event shape to [4, 13]


    def log_prob(self, value):
        # value shape: (*batch_shape, num_agents, num_sub_distributions_per_agent)
        # We need to flatten the value tensor to match the flattened list of distributions for log_prob calculation.
        # The flattened value shape should be (*batch_shape, num_agents * num_sub_distributions_per_agent)

        # Determine batch shape from the value tensor
        batch_shape = value.shape[:-len(self.event_shape)]

        # Flatten the last two dimensions of the value tensor to match the flattened distributions
        # Expected shape after flattening: (*batch_shape, len(self.dists))
        value_flat = value.view(*batch_shape, -1) # Shape (*batch_shape, 52)

        if value_flat.shape[-1] != len(self.dists):
             raise ValueError(f"Value tensor last dimension ({value_flat.shape[-1]}) does not match the number of individual distributions ({len(self.dists)}).")

        # Calculate log_prob for each individual categorical choice
        # dist.log_prob(value_flat[..., i]) will have shape (*batch_shape,)
        # We need to stack these log_probs and then sum them along the last dimension.
        log_probs_list = [dist.log_prob(value_flat[..., i]) for i, dist in enumerate(self.dists)] # List of tensors, each shape (*batch_shape,)

        # Stack the log probabilities along a new dimension
        log_probs_stacked = torch.stack(log_probs_list, dim=-1) # Shape (*batch_shape, len(self.dists))

        # Sum the log probabilities across the individual distributions dimension
        total_log_prob = log_probs_stacked.sum(dim=-1) # Shape (*batch_shape,)

        return total_log_prob

    def sample(self, sample_shape=torch.Size()):
        # Sample from each individual categorical distribution
        # dist.sample(sample_shape) will have shape (sample_shape, *batch_shape)
        # We need to collect samples for each distribution, potentially reshape, and concatenate.

        # Assuming each distribution in self.dists has a batch_shape that is consistent.
        # The batch shape of the MultiCategorical distribution is the batch shape of its constituent distributions.
        # Let's infer the batch shape from the first distribution.
        if not self.dists:
             raise ValueError("Cannot sample from an empty distribution list.")
        dist_batch_shape = self.dists[0].batch_shape

        # Samples from individual distributions, each will have shape (sample_shape, *dist_batch_shape)
        samples_list = [dist.sample(sample_shape) for dist in self.dists]

        # Concatenate the samples along a new dimension
        # The shape after stacking will be (sample_shape, *dist_batch_shape, len(self.dists))
        samples_stacked = torch.stack(samples_list, dim=-1) # Shape (sample_shape, *dist_batch_shape, 52)


        # Reshape the stacked samples to the desired event shape: [num_agents, num_sub_distributions_per_agent]
        # The target shape is (sample_shape, *dist_batch_shape, num_agents, num_sub_distributions_per_agent)
        target_shape = sample_shape + dist_batch_shape + self._event_shape # Concatenate tuples
        samples_reshaped = samples_stacked.view(target_shape) # Shape (sample_shape, *dist_batch_shape, 4, 13)


        return samples_reshaped


    def expand(self, batch_shape, _instance=None):
        # Implement expand if needed for batching
        raise NotImplementedError("Expand not implemented for MultiCategorical.")

    @property
    def batch_shape(self):
        # The batch shape of the MultiCategorical is the batch shape of its constituent distributions.
        if not self.dists:
             return torch.Size([])
        return self.dists[0].batch_shape

    @property
    def event_shape(self):
        return self._event_shape # Return the defined event shape [4, 13]


    def __repr__(self):
        return f"MultiCategorical(num_dists={len(self.dists)}, event_shape={self.event_shape}, batch_shape={self.batch_shape})"

# Define the maker function for the MultiCategorical distribution
# This function will be used by ProbabilisticTensorDictModule to create the distribution
# from the 'logits' output of the network.
# The 'logits' tensor should have shape (*batch_shape, num_agents, num_sub_distributions_per_agent, num_categories_per_sub_action)
# The maker function receives keyword arguments corresponding to the out_keys of the wrapped module.
# In this case, it receives 'logits'.
def multi_categorical_maker(**kwargs):
    # kwargs should contain 'logits' with shape (*batch_shape, num_agents, num_sub_distributions_per_agent, num_categories_per_sub_action)
    logits = kwargs.get("logits")
    if logits is None:
         raise ValueError("MultiCategorical maker function expects 'logits' as input.")

    # Determine batch shape from the logits tensor
    # The dimensions corresponding to num_agents, num_sub_distributions_per_agent, and num_categories_per_sub_action are event/parameter dimensions.
    # The batch shape is the leading dimensions.
    num_categories_per_sub_action = logits.shape[-1] # Should be 3
    num_sub_distributions_per_agent = logits.shape[-2] # Should be 13
    num_agents = logits.shape[-3] # Should be 4
    batch_shape = logits.shape[:-3]

    # Flatten the logits to create individual Categorical distributions
    # The flattened logits shape should be (*batch_shape, num_agents * num_sub_distributions_per_agent, num_categories_per_sub_action)
    logits_flat = logits.view(*batch_shape, -1, num_categories_per_sub_action) # Shape (*batch_shape, 52, 3)

    # Create a list of Categorical distributions
    dists_list = [Categorical(logits=logits_flat[..., i, :]) for i in range(logits_flat.shape[-2])] # List of 52 Categorical distributions, each with batch_shape (*batch_shape,)

    # Instantiate the MultiCategorical distribution
    return MultiCategorical(dists_list)

# This is the original maker function signature provided by the user, which is incomplete.
# It seems the user intended to define the maker function that takes 'nvec'
# and potentially returns a callable that takes 'logits'.
# Let's redefine the maker function to match the expected signature for a distribution_class in ProbabilisticTensorDictModule,
# which is usually a callable that takes the output of the wrapped module (logits in this case) as keyword arguments.
# The initial definition above follows this pattern.

# If the user specifically intended the maker to take 'nvec' first, we would need
# to change how ProbabilisticTensorDictModule is used or define a different type of maker.
# Given the context of using it with ProbabilisticTensorDictModule and the error
# happening during policy call, the maker needs to be the function that takes logits.

# Let's keep the correct multi_categorical_maker defined above and assume
# that the user's provided snippet was an attempt to redefine it incorrectly.
# No changes needed to the correct definition.
