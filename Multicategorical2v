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

    def log_prob(self, value):
        if value.shape[-1] != len(self.dists):
             # Adjust validation for the shape of the action value
             # Expected value shape: (*batch_shape, num_agents, num_actions_per_feature)
             # The total number of individual actions per graph/environment is num_agents * num_sub_distributions
             # The value tensor from the environment step is likely shaped [batch_size, num_agents, num_sub_distributions]
             # We need to match the number of individual distributions (num_agents * num_sub_distributions)
             # with the last dimension of the value tensor after potentially reshaping or handling it.
             # Let's assume the value tensor passed to log_prob is flattened to [batch_size, num_agents * num_sub_distributions]
             # if the distribution samples a flattened action.
             # Revisit this based on the actual output shape of the distribution's sample method.
             # The sample method outputs [batch_size, num_agents * num_sub_distributions] if MultiCategorical is over flattened actions.
             # So, value shape for log_prob should match this.
             # The number of individual distributions is len(self.dists) = num_agents * num_sub_distributions.
             # The last dimension of value should be num_agents * num_sub_distributions.
             # The current check value.shape[-1] != len(self.dists) is correct if value is flattened to [batch_size, num_agents * num_sub_distributions]

             raise ValueError(f"Value shape {value.shape} last dimension must match number of distributions {len(self.dists)}")

        ans = []
        # Ensure splitting happens on the last dimension where individual actions are stacked
        # Value shape is expected to be (batch_size, num_agents * num_sub_distributions)
        # Split along the last dimension (num_agents * num_sub_distributions) to get actions for each individual distribution
        # The split size should be 1 if each individual distribution corresponds to a single action component.
        # If nvec was [3, 3, ..., 3] (13 times), and len(dists) = num_agents * num_sub_distributions = 5 * 13 = 65
        # Value shape is [batch_size, 65]. Splitting with size 1 along dim -1 gives 65 tensors of shape [batch_size, 1].
        values_split = torch.split(value, 1, dim=-1) # Split size 1 along the last dimension
        if len(values_split) != len(self.dists):
             raise ValueError(f"Splitting value tensor resulted in {len(values_split)} tensors, but expected {len(self.dists)}")

        for d, v in zip(self.dists, values_split):
            # Squeeze the last dimension of v, which is 1 after splitting
            ans.append(d.log_prob(v.squeeze(-1))) # log_prob expects shape [batch_size]

        # Stack the log_probs and sum across the individual distributions
        return torch.stack(ans, dim=-1).sum(dim=-1) # Sum log_probs across the individual distributions dimension


    def entropy(self):
        # Stack the entropies and sum across the individual distributions
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1) # Sum entropies across individual distributions dimension


    def sample(self, sample_shape=torch.Size()):
        # Sample from each categorical distribution and stack the results
        # Expected output shape: (*batch_shape, num_agents * num_sub_distributions)
        samples = [d.sample(sample_shape) for d in self.dists]
        return torch.stack(samples, dim=-1) # Stack along a new last dimension


def multi_categorical_maker(nvec):
    # nvec: List of number of categories for each individual action component.
    # Expected length: num_agents * num_sub_distributions_per_agent.
    # Expected elements: num_categories_per_sub_distribution.

    # We need access to num_agents to infer batch size. Assuming it's available globally from env.num_agents.
    # We also need num_sub_distributions_per_agent and num_categories_per_sub_distribution for reshaping.
    # Assuming these are available globally from previous calculations in the main script.
    # Accessing global variables directly within the maker function might not be robust.
    # A better approach is to pass these dimensions when calling multi_categorical_maker in the main script,
    # and have the inner function closure over them.

    # Let's assume num_agents, num_sub_distributions, num_categories are available in the scope
    # where multi_categorical_maker is called and are captured by the closure.
    # This requires modifying how multi_categorical_maker is called in the main script.
    # In the main script, we calculated num_agents, num_sub_distributions, num_categories
    # and nvec_for_maker. We should pass these to multi_categorical_maker.

    # Modified multi_categorical_maker signature and inner function:
    # def multi_categorical_maker(num_agents, num_sub_distributions, num_categories):
    #     nvec = [num_categories] * (num_agents * num_sub_distributions) # Generate nvec here

    #     def get_multi_categorical(logits):
    #         # ... reshaping logic using num_agents, num_sub_distributions, num_categories ...
    #         # ... slicing using nvec ...
    #         # ... return MultiCategorical(dists) ...
    #     return get_multi_categorical

    # For now, let's stick to the user's provided signature `def multi_categorical_maker(nvec):`
    # and assume num_agents, num_sub_distributions, num_categories are available globally
    # or derived from nvec.

    # Assuming nvec is [num_categories] repeated num_agents * num_sub_distributions times.
    num_individual_distributions = len(nvec) # num_agents * num_sub_distributions
    num_categories = nvec[0] if nvec else 0 # Assuming uniform categories

    # Need num_agents and num_sub_distributions.
    # num_individual_distributions = num_agents * num_sub_distributions
    # Cannot uniquely determine num_agents and num_sub_distributions from their product.
    # This suggests multi_categorical_maker needs more information than just nvec,
    # or the structure of nvec needs to implicitly encode num_agents and num_sub_distributions.

    # Let's revert to the assumption that num_agents is available globally
    # and derive num_sub_distributions from num_individual_distributions and num_agents.
    # Assuming num_agents is available globally (like env.num_agents).
    global env # Access global environment to get num_agents
    if env is None:
         print("Warning: Environment not available in multi_categorical_maker scope. Cannot determine num_agents.")
         # Fallback or error handling if env is not available
         # For now, let's assume env is available and num_agents can be accessed.
         num_agents = env.num_agents if 'env' in globals() and env is not None else None
    else:
         num_agents = env.num_agents

    if num_agents is None:
         raise ValueError("num_agents is not available in multi_categorical_maker scope.")

    if num_individual_distributions % num_agents != 0:
         raise ValueError(f"Number of individual distributions ({num_individual_distributions}) is not divisible by num_agents ({num_agents}).")

    num_sub_distributions = num_individual_distributions // num_agents


    def get_multi_categorical(logits):
        # Input logits shape: [batch_size * num_agents, num_sub_distributions * num_categories]

        input_shape = logits.shape
        batch_size_nodes = input_shape[0] # batch_size * num_agents
        params_per_agent = input_shape[1] # num_sub_distributions * num_categories

        # Validate input dimensions against derived dimensions
        if batch_size_nodes % num_agents != 0:
             raise ValueError(f"Input batch size ({batch_size_nodes}) is not divisible by num_agents ({num_agents}).")

        inferred_batch_size = batch_size_nodes // num_agents

        num_individual_distributions_from_logits = params_per_agent // num_categories # Should be num_sub_distributions
        if num_individual_distributions_from_logits != num_sub_distributions:
             raise ValueError(f"Derived individual distributions from logits ({num_individual_distributions_from_logits}) does not match calculated num_sub_distributions ({num_sub_distributions}).")


        # Reshape logits from [batch_size * num_agents, num_sub_distributions * num_categories]
        # to [batch_size, num_agents * num_sub_distributions, num_categories] for slicing.
        # This is equivalent to reshaping to [inferred_batch_size, num_individual_distributions, num_categories].

        reshaped_logits = logits.view(
            inferred_batch_size,
            num_individual_distributions, # num_agents * num_sub_distributions
            num_categories # num_categories
        )

        dists = []
        # Slice along the second dimension (num_individual_distributions) to get parameters for each distribution
        for i in range(num_individual_distributions):
            # Slice the parameters for the i-th individual distribution
            # The slice should take all batch elements and all categories for this distribution.
            # Assuming uniform categories for simpler slicing.
            dist_logits = reshaped_logits[:, i, :] # Shape [batch_size, num_categories]
            dists.append(Categorical(logits=dist_logits))

        # Ensure the MultiCategorical class is accessible here (imported at the top)
        return MultiCategorical(dists)

    return get_multi_categorical

# Assuming MultiCategorical class is defined above or imported.
# If it's defined in the same file, ensure its definition is before multi_categorical_maker.
