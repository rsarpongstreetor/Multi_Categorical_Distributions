# mypy: allow-untyped-defs
from typing import Optional, Union, List

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
# from torch.distributions.utils import broadcast_all # Not needed for MultiCategorical in this structure
from torch.distributions.categorical import Categorical as TorchCategorical
from torchrl.data import CompositeSpec # Assuming CompositeSpec is available

# --- Define Custom MultiCategorical Distribution (Inheriting from Distribution) ---
# Reverting to direct inheritance from Distribution as requested.

class MultiCategorical(Distribution):
    """
    A MultiCategorical distribution for multiple agents with multiple independent categorical action features.

    Args:
        logits (torch.Tensor): The logits tensor.
            Expected shape: `[..., num_agents, num_individual_actions_features, num_action_categories]`
        action_spec (CompositeSpec): The action specification of the environment.
            Used to infer num_agents, num_individual_actions_features, and dtype.
    """
    arg_constraints = {"logits": constraints.real} # Define argument constraints if necessary
    has_enumerate_support = False # Enumerating support for multi-categorical is complex and likely not needed

    def __init__(self, logits: torch.Tensor, action_spec: CompositeSpec, validate_args: Optional[bool] = None):
        # print("MultiCategorical.__init__ called.") # Debug print
        # Logits shape is expected to be [..., num_agents, num_individual_actions_features, num_action_categories]
        self.logits = logits
        self.action_spec = action_spec

        # Infer shapes from action_spec
        # action_spec is expected to be a CompositeSpec where the relevant part is a DiscreteTensorSpec
        # with shape [num_agents, num_individual_actions_features]
        action_spec_shape = self.action_spec.shape # Expected [num_agents, num_individual_actions_features] for unbatched

        if len(action_spec_shape) < 2:
             raise ValueError(f"Unexpected action_spec shape: {action_spec_shape}. Expected at least 2 dimensions (num_agents, num_individual_actions_features).")

        self.num_agents = action_spec_shape[0]
        self.num_individual_actions_features = action_spec_shape[1]


        # Infer num_action_categories from the logits shape
        # Logits shape is expected to be [..., num_agents, num_individual_actions_features, num_action_categories]
        if logits.ndim < 3:
             raise ValueError(f"Unexpected logits shape: {logits.shape}. Expected at least 3 dimensions.")

        num_action_categories = logits.shape[-1]
        self.num_action_categories = num_action_categories


        # Validate the shape of the input logits tensor based on inferred shapes
        expected_shape_suffix = (self.num_agents, self.num_individual_actions_features, self.num_action_categories)
        if self.logits.shape[-len(expected_shape_suffix):] != expected_shape_suffix:
             raise ValueError(f"Logits shape mismatch with inferred shapes. Expected shape ending with {expected_shape_suffix}, but got {self.logits.shape}. Full logits shape: {self.logits.shape}")

        # The batch shape of this distribution is the batch shape of the input logits (excluding the last 3 dims).
        batch_shape = self.logits.shape[:-3] # Exclude num_agents, num_individual_actions_features, num_action_categories

        # The event shape is [num_agents, num_individual_actions_features]
        event_shape = torch.Size([self.num_agents, self.num_individual_actions_features])

        super(MultiCategorical, self).__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

        # Store dtype for deterministic sample from the action spec
        # Assuming the action spec has a dtype attribute, likely from the DiscreteTensorSpec
        self._action_spec_dtype = action_spec.dtype if hasattr(action_spec, 'dtype') else torch.long


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultiCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        # Expand logits to the new batch shape
        new.logits = self.logits.expand(batch_shape + self.event_shape + (self.num_action_categories,)) # Expand batch and keep event/category dims
        new.action_spec = self.action_spec # Action spec doesn't change with batch expansion

        new.num_agents = self.num_agents # Keep original inferred values
        new.num_individual_actions_features = self.num_individual_actions_features
        new.num_action_categories = self.num_action_categories
        new._action_spec_dtype = self._action_spec_dtype

        super(MultiCategorical, new).__init__(batch_shape, new.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self.logits.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=2) # Event dim is 2 for [num_agents, num_individual_actions_features]
    def support(self):
        # Support for MultiCategorical is a product of individual categorical supports.
        # This is complex to represent concisely. Returning a general constraint.
        # Each element in the event shape [num_agents, num_individual_actions_features]
        # has a support of {0, 1, ..., num_action_categories - 1}.
        # For simplicity and alignment with torch.distributions, we might not define a specific support constraint
        # if it's too complex. However, if a constraint is required, it would be something
        # indicating that each element is a non-negative integer less than num_action_categories.
        # For now, we'll rely on validation in log_prob if validate_args is True.
        raise NotImplementedError("Support for MultiCategorical is complex and not explicitly constrained.")


    @property
    def probs(self) -> torch.Tensor:
        """
        Returns the probabilities derived from the logits.
        Shape: `[..., num_agents, num_individual_actions_features, num_action_categories]`
        """
        return torch.softmax(self.logits, dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """
        Returns the mode of the distribution as a stand-in for mean for deterministic evaluation.
        This method is explicitly defined to satisfy torchrl's requirements.
        """
        # print("MultiCategorical.mean property accessed.") # Debug print
        # The mode of a Categorical distribution is the index of the highest logit.
        # self.logits has shape [..., num_agents, num_individual_actions_features, num_action_categories]
        # We need to take argmax over the last dimension.
        mode = torch.argmax(self.logits, dim=-1) # Shape [..., num_agents, num_individual_actions_features]
        return mode.to(self._action_spec_dtype) # Ensure dtype matches action spec


    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the MultiCategorical distribution.
        This corresponds to the action with the highest probability for each
        of the individual categorical choices.
        """
        # print("MultiCategorical.mode property accessed.") # Debug print
        # The mode is already computed in the mean property for this structure.
        return self.mean # Use the mean property which is the mode

    @property
    def variance(self) -> Tensor:
         # Variance for MultiCategorical is complex as it's a product distribution.
         # Variance of a categorical is p * (1-p). For a product, it's the product of variances
         # for the log_prob (sum of log_probs).
         # However, the variance of the *sample* itself is more complex.
         # It might be better to not implement this if not strictly needed.
         raise NotImplementedError("Variance for MultiCategorical is not implemented.")


    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape batch of samples.

        Returns:
            torch.Tensor: The sampled actions.
                Shape: `[sample_shape, ..., num_agents, num_individual_actions_features]`
        """
        shape = self._extended_shape(sample_shape)
        # We need to sample independently for each agent and each action feature.
        # The logits are already structured for this:
        # [..., num_agents, num_individual_actions_features, num_action_categories]
        # We can use torch.distributions.Categorical directly on the last dimension.

        # Create a base Categorical distribution for sampling.
        # Need to reshape logits to [..., num_action_categories] for the base Categorical.
        # The batch shape for this temporary Categorical will be
        # [..., num_agents, num_individual_actions_features]
        original_shape = self.logits.shape
        reshaped_logits = self.logits.view(-1, self.num_action_categories) # Flatten batch and event dims

        # Create a temporary Categorical distribution
        base_categorical = TorchCategorical(logits=reshaped_logits)

        # Sample from the base categorical distribution
        # The sample method of the base categorical returns a tensor with shape:
        # [sample_shape, batch_shape of base_categorical]
        # In our case: [sample_shape, flattened_batch_and_event_dims]
        # where flattened_batch_and_event_dims is product of batch shape and event shape

        flattened_batch_and_event_dims = torch.Size(original_shape[:-1]).numel()
        sample_batch_shape = sample_shape + torch.Size([flattened_batch_and_event_dims])

        sampled_flat = base_categorical.sample(sample_shape) # Shape: [sample_shape, flattened_batch_and_event_dims]

        # Reshape the sampled actions back to the desired shape
        # Desired shape: [sample_shape, ..., num_agents, num_individual_actions_features]
        output_shape = sample_shape + self.batch_shape + self.event_shape
        sampled_action = sampled_flat.view(output_shape)

        return sampled_action.to(self._action_spec_dtype)


    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density of a given sample.

        Args:
            value (torch.Tensor): The action tensor for which to compute the log probability.
                Expected shape: `[..., num_agents, num_individual_actions_features]`

        Returns:
            torch.Tensor: The log probability of the action.
                Shape: `[..., 1]` (summed log probs over action features and agents)
        """
        if self._validate_args:
             self._validate_sample(value)

        # We need to compute the log probability for each individual categorical choice
        # and then sum them up.
        # The value tensor has shape [..., num_agents, num_individual_actions_features]
        # The logits tensor has shape [..., num_agents, num_individual_actions_features, num_action_categories]

        # Create a base Categorical distribution for log_prob calculation.
        # Reshape logits to [..., num_action_categories] for the base Categorical.
        # The batch shape for this temporary Categorical will be
        # [..., num_agents, num_individual_actions_features]
        original_logits_shape = self.logits.shape
        reshaped_logits = self.logits.view(-1, self.num_action_categories) # Flatten batch and event dims

        # Create a temporary Categorical distribution
        base_categorical = TorchCategorical(logits=reshaped_logits)

        # Reshape the value tensor to match the batch shape of the base Categorical
        # Value shape: [..., num_agents, num_individual_actions_features]
        # Desired shape for base_categorical.log_prob: [..., flattened_batch_and_event_dims]
        original_value_shape = value.shape
        flattened_batch_and_event_dims = torch.Size(original_value_shape).numel() // self.event_shape.numel() * self.event_shape.numel()
        # Ensure value shape is compatible with logits shape before flattening
        expected_value_suffix = self.event_shape
        if value.shape[-len(expected_value_suffix):] != expected_value_suffix:
             raise ValueError(f"Value shape mismatch with event shape. Expected shape ending with {expected_value_suffix}, but got {value.shape}. Full value shape: {value.shape}")

        # Flatten the value tensor
        reshaped_value = value.view(-1) # Flatten all dimensions

        # The log_prob method of the base categorical expects a value tensor with shape:
        # [..., batch_shape of base_categorical]
        # In our case, after flattening, the batch shape is the product of the original batch shape and event shape.
        # The value tensor should have this flattened shape.
        # The log_prob returns a tensor with the same batch shape: [..., flattened_batch_and_event_dims]
        log_prob_flat = base_categorical.log_prob(reshaped_value) # Shape [flattened_batch_and_event_dims]

        # Reshape the log probabilities back to [..., num_agents, num_individual_actions_features]
        # This will have the same batch shape as the original input tensors (logits and value)
        log_prob_per_choice = log_prob_flat.view(self.batch_shape + self.event_shape) # Shape [..., num_agents, num_individual_actions_features]


        # Sum over the event dimensions ([num_agents, num_individual_actions_features])
        # to get the log probability for the entire multi-categorical action.
        log_prob_summed = log_prob_per_choice.sum(dim=(-1, -2)) # Sum over num_individual_actions_features and num_agents

        # Need to unsqueeze the output to match the expected shape [..., 1]
        return log_prob_summed.unsqueeze(-1)

    def deterministic_sample(self) -> torch.Tensor:
        """
        Returns a deterministic sample from the distribution (the mode).
        This method is explicitly added to satisfy ProbabilisticActor in evaluation mode.
        """
        # print("MultiCategorical.deterministic_sample called.") # Debug print
        return self.mode # Return the mode


# Maker function for ProbabilisticActor
# Re-defining the maker function here to ensure it uses the MultiCategorical defined in this cell.
def multi_categorical_maker(nvec: List[int]):
    """
    A maker function to create the MultiCategorical distribution for ProbabilisticActor.
    This function is called by ProbabilisticActor with dist_kwargs containing 'logits'.
    It also receives the output `spec` of the ProbabilisticActor.
    """
    # The maker function should return a callable that takes the distribution parameters
    # and returns a Distribution instance.
    # The ProbabilisticActor will pass dist_kwargs derived from dist_keys and the output `spec`
    # to this callable.

    def get_multi_categorical_with_spec(spec: CompositeSpec = None, **dist_kwargs) -> MultiCategorical:
        if 'logits' not in dist_kwargs:
            raise ValueError("multi_categorical_maker expects 'logits' in dist_kwargs")
        if spec is None:
             raise ValueError("multi_categorical_maker requires 'spec' argument from ProbabilisticActor")

        logits = dist_kwargs['logits']

        # Extract action spec from the output spec of ProbabilisticActor
        # The output spec has keys corresponding to out_keys: ("agents", "action"), ("agents", "sample_log_prob")
        # The action spec is located at ("agents", "action") within the output spec.
        if ("agents", "action") not in spec.keys(include_nested=True):
             raise ValueError("Action spec not found at ('agents', 'action') in the provided spec")

        action_spec = spec[("agents", "action")] # Get the action spec

        # Now initialize MultiCategorical with logits and the extracted action_spec
        return MultiCategorical(logits=logits, action_spec=action_spec)


    return get_multi_categorical_with_spec # Return the callable that accepts spec and logits
