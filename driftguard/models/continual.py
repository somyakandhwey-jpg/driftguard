"""
models/continual.py -- Phase 4: continual learning without full retraining.

Two complementary mechanisms, used together in training/train_phase4_continual.py:

1. EWC (Elastic Weight Consolidation): after finishing a time chunk, estimate
   how important each weight was (diagonal Fisher information) and penalize
   future updates that move important weights far from their old values.
   This is what lets the model adapt to new (drifted) fraud patterns without
   catastrophically forgetting old ones.

2. ReplayBuffer: keeps a small reservoir sample of past nodes so each new
   training step also sees a bit of old-era data, cheaply reinforcing what
   EWC protects structurally.
"""
import copy
import random
import torch


class EWC:
    """Model-agnostic EWC: works for the plain GNN classifier (Phase 4) and
    the diffusion denoiser (Phase 5), since it never calls the model itself
    -- the caller supplies a zero-argument `loss_fn` closure that runs
    whatever forward pass is appropriate and returns a scalar loss."""

    def __init__(self, model, ewc_lambda=1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher = {}          # name -> Fisher diagonal estimate
        self.optimal_params = {}  # name -> params at end of previous task

    def register_task(self, loss_fn):
        """Call after training on a task/time-chunk to snapshot importance.
        loss_fn: callable() -> scalar loss tensor (built from self.model's
        current parameters), e.g.:
            lambda: F.binary_cross_entropy_with_logits(model(x, ei)[mask], y[mask].float())
        """
        self.model.eval()
        self.model.zero_grad()

        loss = loss_fn()
        loss.backward()

        fisher = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher[name] = param.grad.detach().clone() ** 2
            else:
                fisher[name] = torch.zeros_like(param)

        # accumulate (running average) so importance compounds across many tasks
        for name in fisher:
            if name in self.fisher:
                self.fisher[name] = 0.5 * self.fisher[name] + 0.5 * fisher[name]
            else:
                self.fisher[name] = fisher[name]

        self.optimal_params = {
            name: param.detach().clone() for name, param in self.model.named_parameters()
        }
        self.model.zero_grad()

    def penalty(self):
        """EWC regularization term to add to the current task's loss."""
        if not self.fisher:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.optimal_params[name]) ** 2).sum()
        return self.ewc_lambda * loss


class ReplayBuffer:
    """Reservoir sampling buffer of node indices (and which time-chunk they came
    from), used to mix a little old data back into every new training step."""

    def __init__(self, capacity=2000, seed=0):
        self.capacity = capacity
        self.buffer = []  # list of (global_node_idx, chunk_id)
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, node_indices, chunk_id):
        for idx in node_indices:
            self.seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append((idx, chunk_id))
            else:
                j = self.rng.randint(0, self.seen - 1)
                if j < self.capacity:
                    self.buffer[j] = (idx, chunk_id)

    def sample(self, batch_size):
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        return [idx for idx, _ in self.rng.sample(self.buffer, batch_size)]

    def __len__(self):
        return len(self.buffer)


def clone_model(model):
    return copy.deepcopy(model)
