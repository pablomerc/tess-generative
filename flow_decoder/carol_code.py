import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko
from abc import ABC, abstractmethod
from typing import Tuple, List


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for decoders."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def get_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return a distribution object for sampling and log_prob computation."""
        pass
    
    @abstractmethod
    def sample(self, z: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample from the decoder given latent codes."""
        pass


class NormalizingFlowDecoder(BaseDecoder):
    """Normalizing Flow decoder using zuko."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        flow_type: str = 'maf',
        n_transforms: int = 4,
        hidden_dims: Tuple[int, ...] = (64, 64)
    ):
        super().__init__(input_dim, output_dim)
        if flow_type == 'maf':
            self.flow = zuko.flows.MAF(
                features=output_dim,
                context=input_dim,
                transforms=n_transforms,
                hidden_features=hidden_dims
            )
        elif flow_type == 'nsf':
            self.flow = zuko.flows.NSF(
                features=output_dim,
                context=input_dim,
                transforms=n_transforms,
                hidden_features=hidden_dims
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}. Supported: 'maf', 'nsf'")
    
    def forward(self, z: torch.Tensor) -> torch.distributions.Distribution:
        return self.flow(z)
    
    def sample(self, z: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        dist = self.forward(z)
        if n_samples == 1:
            return dist.sample()
        else:
            return dist.sample((n_samples,))
    
    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dist = self.forward(z)
        return dist.log_prob(x)
    
    def get_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x=x, z=z).mean()


class FlowMatchingDecoder(BaseDecoder):
    """Flow Matching decoder using conditional flow matching."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128,],
        n_integration_steps: int = 100,
        activation: nn.Module = nn.SiLU()
    ):
        super().__init__(input_dim, output_dim)
        self.n_integration_steps = n_integration_steps
        
        # Set up velocity network
        network_dims = [output_dim + 1 + input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(network_dims) - 1):
            layers.append(nn.Linear(network_dims[i], network_dims[i + 1]))
            if i < len(network_dims) - 2:
                layers.append(activation)
        self.vector_field = nn.Sequential(*layers)
    
    def get_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute conditional probability path x_t = t*x1 + (1-t)*x0."""
        # x0 ~ N(0, I)
        x_t = t.unsqueeze(-1) * x1 + (1 - t.unsqueeze(-1)) * x0
        return x_t
    
    def vector_field_forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute vector field v_θ(x, t, z)."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        inputs = torch.cat([x, t.unsqueeze(-1), z], dim=-1)
        return self.vector_field(inputs)
    
    def compute_cfm_loss(self, x1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute Conditional Flow Matching loss."""
        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=x1.device)
        x_t = self.get_conditional_flow(x0=x0, x1=x1, t=t)
        v_target = x1 - x0
        v_pred = self.vector_field_forward(x_t, t, z)
        loss = F.mse_loss(v_pred, v_target)
        return loss
    
    def sample_ode(self, z: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample by solving ODE from noise to data."""
        batch_size = z.shape[0] if n_samples == 1 else n_samples * z.shape[0]
        device = z.device
        x = torch.randn(batch_size, self.output_dim, device=device)
        
        if n_samples > 1:
            z_expanded = z.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, z.shape[-1])
        else:
            z_expanded = z
        
        dt = 1.0 / self.n_integration_steps
        for i in range(self.n_integration_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            v = self.vector_field_forward(x, t, z_expanded)
            x = x + dt * v
        
        if n_samples > 1:
            x = x.view(n_samples, z.shape[0], self.output_dim)
        return x
    
    def sample(self, z: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        return self.sample_ode(z, n_samples)
    
    def get_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.compute_cfm_loss(x1=x, z=z)