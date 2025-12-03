import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import zuko
import math

class BaseDecoder(nn.Module, ABC):
    '''Abstract base class for decoders.'''
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    @abstractmethod
    def get_loss(self, x:torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        '''Return a distribution object for sampling and log_prob computation.'''
        pass
    @abstractmethod
    def sample(self, z: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        '''Sample from the decoder given latent codes.'''
        pass


class VelocityField(nn.Module, ABC):
    """Abstract base class for velocity field networks"""

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field v_θ(x, t, z)
        Args:
            - x: position (batch_size, output_dim)
            - t: time (batch_size,) or scalar
            - z: conditioning (batch_size, input_dim)
        Returns:
            - velocity: (batch_size, output_dim)
        """
        pass


class MLPVelocityField(VelocityField):
    """Simple MLP velocity field (original implementation)"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128], activation: nn.Module = nn.SiLU()):
        super().__init__()
        network_dims = [output_dim + 1 + input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(network_dims) - 1):
            layers.append(nn.Linear(network_dims[i], network_dims[i + 1]))
            if i < len(network_dims) - 2:
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        inputs = torch.cat([x, t.unsqueeze(-1), z], dim=-1)
        return self.network(inputs)


class FourierEncoder(nn.Module):
    """Time embedding using Fourier features"""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        # Initialize with smaller weights for better stability
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (batch_size,)
        Returns:
        - embeddings: (batch_size, dim)
        """
        t = t.view(-1, 1)  # (batch_size, 1)
        t = torch.clamp(t, 0.0, 1.0)

        freqs = t * self.weights * 2 * math.pi  # (batch_size, half_dim)
        sin_embed = torch.sin(freqs)  # (batch_size, half_dim)
        cos_embed = torch.cos(freqs)  # (batch_size, half_dim)
        result = torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (batch_size, dim)

        # Check for NaN/Inf and replace if necessary
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("WARNING: NaN/Inf in FourierEncoder. Replaced with zeros.")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

        return result


class ResidualLayer(nn.Module):
    """Residual block for U-Net"""
    def __init__(self, channels: int, time_embed_dim: int, z_embed_dim: int, film: bool = True, unconditional: bool = False):
        super().__init__()
        self.film_bool = film
        self.unconditional = unconditional
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        if self.film_bool:
            self.film_layer = FiLMLayer(channels, time_embed_dim + z_embed_dim)
            # For unconditional models, initialize FiLM layer to produce identity (scale=0, shift=0)
            if self.unconditional:
                with torch.no_grad():
                    # Initialize the last layer to produce zeros (scale=0, shift=0 means x*1 + 0 = x)
                    if self.film_layer.scale_shift.bias is not None:
                        self.film_layer.scale_shift.bias.zero_()
        else:
            # Time adapter
            self.time_adapter = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, channels)
            )
            # Conditioning adapter
            self.z_adapter = nn.Sequential(
                nn.Linear(z_embed_dim, z_embed_dim),
                nn.SiLU(),
                nn.Linear(z_embed_dim, channels)
            )
            # For unconditional models, initialize z_adapter biases to zero
            if self.unconditional:
                with torch.no_grad():
                    # First linear layer
                    if self.z_adapter[0].bias is not None:
                        self.z_adapter[0].bias.zero_()
                    # Second linear layer
                    if self.z_adapter[2].bias is not None:
                        self.z_adapter[2].bias.zero_()

    #     # Initialize weights for better stability
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     """Initialize weights for better stability"""
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
    #         elif isinstance(module, nn.Linear):
    #             nn.init.xavier_normal_(module.weight, gain=0.1)  # Smaller gain
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (batch_size, c, h, w)
        - t_embed: (batch_size, t_embed_dim)
        - z_embed: (batch_size, z_embed_dim)
        """
        res = x.clone()

        # Initial conv block
        x = self.block1(x)

        # # Clamp to prevent extreme values
        # x = torch.clamp(x, -10.0, 10.0)

        if self.film_bool:
            condition = torch.cat([t_embed, z_embed], dim=1)
            x = self.film_layer(x, condition)
        else:
            # Add time embedding with scaling
            t_embed_out = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)
            x = x + t_embed_out

            # Add conditioning embedding with scaling
            z_embed_out = self.z_adapter(z_embed).unsqueeze(-1).unsqueeze(-1)
            x = x + z_embed_out

        # Second conv block
        x = self.block2(x)

        # Add back residual
        x = x + res

        return x


class Encoder(nn.Module):
    """U-Net encoder block"""
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int,
                 t_embed_dim: int, z_embed_dim: int, use_film: bool = True, unconditional: bool = False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, z_embed_dim, film=use_film, unconditional=unconditional)
            for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, z_embed)

        # Downsample
        x = self.downsample(x)
        return x


class Midcoder(nn.Module):
    """U-Net middle block"""
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int,
                 z_embed_dim: int, use_film: bool = True, unconditional: bool = False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, z_embed_dim, film=use_film, unconditional=unconditional)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed, z_embed)
        return x


class Decoder(nn.Module):
    """U-Net decoder block"""
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int,
                 t_embed_dim: int, z_embed_dim: int, use_film: bool = True, unconditional: bool = False):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        )
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, z_embed_dim, film=use_film, unconditional=unconditional)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, z_embed)

        return x


class UNetVelocityField(VelocityField):
    """U-Net velocity field for image-like data"""

    def __init__(self, input_dim: int, output_dim: int, image_size: int = 28,
                 channels: List[int] = [32, 64, 128], num_residual_layers: int = 2,
                 t_embed_dim: int = 40, z_embed_dim: int = 40, use_film: bool = True,
                 unconditional: bool = False, num_channels: int = 1):
        super().__init__()

        self.image_size = image_size
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.use_film = use_film
        self.unconditional = unconditional or (input_dim == 0)

        # Verify output_dim matches image dimensions
        expected_dim = num_channels * image_size * image_size
        if output_dim != expected_dim:
            raise ValueError(f"output_dim {output_dim} must equal num_channels * image_size² = {expected_dim}")

        # Initial convolution: (batch_size, num_channels, image_size, image_size) -> (batch_size, c_0, image_size, image_size)
        self.init_conv = nn.Sequential(
            nn.Conv2d(num_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize z embedder (conditioning)
        # For unconditional models, we still need the network structure but will initialize biases to zero
        effective_input_dim = max(input_dim, 1)
        self.z_embedder = nn.Sequential(
            nn.Linear(effective_input_dim, z_embed_dim),
            nn.SiLU(),
            nn.Linear(z_embed_dim, z_embed_dim)
        )

        # For unconditional models, initialize z_embedder biases to zero so zeros produce zero output
        if self.unconditional:
            with torch.no_grad():
                # First linear layer: set bias to zero
                if self.z_embedder[0].bias is not None:
                    self.z_embedder[0].bias.zero_()
                # Second linear layer: set bias to zero
                if self.z_embedder[2].bias is not None:
                    self.z_embedder[2].bias.zero_()

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, z_embed_dim,
                                    use_film=use_film, unconditional=self.unconditional))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, z_embed_dim,
                                    use_film=use_film, unconditional=self.unconditional))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, z_embed_dim,
                                 use_film=use_film, unconditional=self.unconditional)

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], num_channels, kernel_size=3, padding=1)

    #     # Initialize weights for stability
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     """Initialize weights for better stability"""
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
    #         elif isinstance(module, nn.Linear):
    #             nn.init.xavier_normal_(module.weight, gain=0.1)  # Smaller gain for stability
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
    #         elif isinstance(module, nn.BatchNorm2d):
    #             nn.init.constant_(module.weight, 1)
    #             nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (batch_size, output_dim) - flattened image
        - t: (batch_size,) or scalar - time
        - z: (batch_size, input_dim) - conditioning
        Returns:
        - velocity: (batch_size, output_dim) - flattened velocity field
        """
        batch_size = x.shape[0]

        # Reshape x to image format
        x = x.view(batch_size, self.num_channels, self.image_size, self.image_size)

        # Handle time dimension
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.shape[0] != batch_size:
            t = t.expand(batch_size)

        # Embed t and z with clamping
        t_embed = self.time_embedder(t)  # (batch_size, t_embed_dim)
        z_embed = self.z_embedder(z)     # (batch_size, z_embed_dim)


        # Initial convolution
        x = self.init_conv(x)  # (batch_size, c_0, image_size, image_size)
        # x = torch.clamp(x, -5.0, 5.0)

        residuals = []

        # Encoders
        for i, encoder in enumerate(self.encoders):
            x = encoder(x, t_embed, z_embed)
            # x = torch.clamp(x, -5.0, 5.0)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, z_embed)
        # x = torch.clamp(x, -5.0, 5.0)

        # Decoders
        for i, decoder in enumerate(self.decoders):
            res = residuals.pop()
            x = x + res  # residual connection
            x = decoder(x, t_embed, z_embed)
            # x = torch.clamp(x, -5.0, 5.0)

        # Final convolution
        x = self.final_conv(x)  # (batch_size, num_channels, image_size, image_size)
        # x = torch.clamp(x, -3.0, 3.0)

        # Flatten back to original format
        x = x.view(batch_size, self.output_dim)

        return x


class FlowMatchingDecoder(BaseDecoder):
    '''Flow Matching decoder using conditional flow matching with configurable velocity field.

    Can be used conditionally (with z) or unconditionally (z=None or fixed z).

    Examples:
        # Conditional usage:
        decoder = FlowMatchingDecoder(input_dim=64, output_dim=784)
        loss = decoder.get_loss(x, z=latent_codes)
        samples = decoder.sample(z=latent_codes)

        # Unconditional usage:
        decoder = FlowMatchingDecoder(input_dim=0, output_dim=784, unconditional=True)
        loss = decoder.get_loss(x, z=None)
        samples = decoder.sample(z=None, batch_size=32, device=device)
    '''

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        velocity_field_type: str = "mlp",
        hidden_dims: List[int] = [128, 128],
        n_integration_steps: int = 100,
        activation: nn.Module = nn.SiLU(),
        # U-Net specific parameters
        image_size: int = 28,
        unet_channels: List[int] = [32, 64, 128],
        num_residual_layers: int = 2,
        t_embed_dim: int = 40,
        z_embed_dim: int = 40,
        use_film: bool = True,
        unconditional: bool = False,
        num_channels: int = 1
    ):
        super().__init__(input_dim, output_dim)
        self.n_integration_steps = n_integration_steps
        self.use_film = use_film
        self.unconditional = unconditional or (input_dim == 0)

        # For unconditional models with input_dim=0, we use 1 for network structure
        # but will pass zeros. For conditional models, use actual input_dim.
        effective_input_dim = max(input_dim, 1)

        # Initialize velocity field based on type
        if velocity_field_type == "mlp":
            self.vector_field = MLPVelocityField(effective_input_dim, output_dim, hidden_dims, activation)
        elif velocity_field_type == "unet":
            self.vector_field = UNetVelocityField(
                effective_input_dim, output_dim, image_size, unet_channels,
                num_residual_layers, t_embed_dim, z_embed_dim,
                use_film=use_film, unconditional=self.unconditional, num_channels=num_channels
            )
        else:
            raise ValueError(f"Unknown velocity_field_type: {velocity_field_type}. Supported: 'mlp', 'unet'")

    def _get_z(self, z: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        '''Get conditioning vector, using zeros if unconditional or z is None.'''
        if self.unconditional or z is None:
            # Return zeros for unconditional generation
            # Use effective_input_dim (at least 1) for network compatibility
            effective_dim = max(self.input_dim, 1)
            return torch.zeros(batch_size, effective_dim, device=device)
        return z

    def get_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''Compute conditional probability path x_t = t*x1 + (1-t)*x0.'''
        # x0 ~ N(0, I)
        x_t = t.unsqueeze(-1) * x1 + (1 - t.unsqueeze(-1)) * x0
        return x_t

    def vector_field_forward(self, x: torch.Tensor, t: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Compute vector field v_θ(x, t, z). If z is None, uses zeros for unconditional.'''
        batch_size = x.shape[0]
        device = x.device
        z = self._get_z(z, batch_size, device)
        return self.vector_field(x, t, z)

    def compute_cfm_loss(self, x1: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Compute Conditional Flow Matching loss. If z is None, uses zeros for unconditional.'''
        batch_size = x1.shape[0]
        device = x1.device
        z = self._get_z(z, batch_size, device)

        # Clamp input to prevent extreme values
        # x1 = torch.clamp(x1, -3.0, 3.0)
        # z = torch.clamp(z, -5.0, 5.0)

        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device)
        x_t = self.get_conditional_flow(x0=x0, x1=x1, t=t)
        v_target = x1 - x0

        # Clip targets to prevent extreme values
        # v_target = torch.clamp(v_target, -5.0, 5.0)

        v_pred = self.vector_field_forward(x_t, t, z)
        # v_pred = torch.clamp(v_pred, -5.0, 5.0)

        loss = F.mse_loss(v_pred, v_target)

        return loss

    def sample_ode(self, z: Optional[torch.Tensor] = None, n_samples: int = 1,
                   batch_size: Optional[int] = None, device: Optional[torch.device] = None,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        '''Sample by solving ODE from noise to data.

        For unconditional: pass z=None and specify batch_size and device.
        For conditional: pass z tensor.
        '''
        if z is None:
            # Unconditional sampling
            if batch_size is None or device is None:
                raise ValueError("For unconditional sampling, must provide batch_size and device")
            z = self._get_z(None, batch_size, device)

        actual_batch_size = z.shape[0] if n_samples == 1 else n_samples * z.shape[0]
        actual_device = z.device

        x = torch.randn(actual_batch_size, self.output_dim, device=actual_device, generator=generator)
        if n_samples > 1:
            z_expanded = z.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, z.shape[-1])
        else:
            z_expanded = z

        dt = 1.0 / self.n_integration_steps
        for i in range(self.n_integration_steps):
            t = torch.full((actual_batch_size,), i * dt, device=actual_device)
            v = self.vector_field_forward(x, t, z_expanded)
            # Use softer clamping or remove entirely - hard clamps can cause artifacts
            # v = torch.clamp(v, -10.0, 10.0) # Clamp velocity
            x = x + dt * v
            # Remove hard clamp on position - let the model learn the natural data range
            # x = torch.clamp(x, -3.0, 3.0) # Clamp position

        if n_samples > 1:
            x = x.view(n_samples, z.shape[0], self.output_dim)
        return x

    def sample(self, z: Optional[torch.Tensor] = None, n_samples: int = 1,
               batch_size: Optional[int] = None, device: Optional[torch.device] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        '''Sample from the decoder.

        For unconditional: sample(z=None, batch_size=32, device=device)
        For conditional: sample(z=latent_tensor)
        '''
        return self.sample_ode(z, n_samples, batch_size, device, generator)

    def get_loss(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Compute loss. For unconditional, pass z=None.'''
        return self.compute_cfm_loss(x1=x, z=z)


class NormalizingFlowDecoder(BaseDecoder):
    '''Normalizing Flow decoder using zuko.'''
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



class FiLMLayer(nn.Module):
    "Feature-wise Linear Modulation layer"
    def __init__(self,num_channels,condition_dim):
        super().__init__()
        self.scale_shift=nn.Linear(condition_dim, num_channels*2)
        hidden=max(condition_dim,num_channels)
        # self.scale_shift = nn.Sequential(
        #         nn.Linear(condition_dim, hidden),
        #         nn.SiLU(),
        #         nn.Linear(hidden, num_channels*2)
        #     )


    def forward(self, x, condition):
        scale_shift=self.scale_shift(condition)
        scale,shift=scale_shift.chunk(2,dim=-1)

        expand_dims = x.dim() - scale.dim()
        if expand_dims < 0:
            raise ValueError(
                f"Condition tensor has higher rank than input (x.dim()={x.dim()}, condition.dim()={scale.dim()})"
            )
        if expand_dims > 0:
            view_shape = scale.shape + (1,) * expand_dims
            scale = scale.view(*view_shape)
            shift = shift.view(*view_shape)

        return x*(1+scale)+shift
