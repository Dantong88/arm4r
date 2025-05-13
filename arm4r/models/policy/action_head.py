import abc
from typing import Dict, Union, Optional, Literal 
import torch
import torch.nn as nn
import torch.distributions as D
from timm.layers import Mlp
from torch.nn import functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from arm4r.models.policy.utils import ConditionedDiffusion

class PredHead(abc.ABC, nn.Module):
    """
    Abstract class for prediction head
    """
    @abc.abstractmethod
    def forward(self, x : torch.Tensor) -> Union[torch.Tensor, D.Distribution]:
        """
        Forward pass of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            Union[torch.Tensor, D.Distribution], (B, output_dim) or D.Distribution
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the prediction head
        Args:
            x: torch.Tensor, (B, input_dim)
            y: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, scalar
        """
        raise NotImplementedError

class MLPHead(PredHead):
    """
    a 2 layer mlp
    """
    def __init__(
        self, 
        input_dim : int, 
        hidden_features : int,
        output_dim : int,
        loss_fn : nn.Module,
    ): 
        super(MLPHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_features = hidden_features
        self.output_dim = output_dim

        self.mlp = Mlp(
            in_features=input_dim, 
            hidden_features=hidden_features, 
            out_features=output_dim
        )
        self.loss_fn = loss_fn
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.mlp(x)

    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the MLP
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.forward(x)
    
    def loss(self, latent : torch.Tensor, gt_action : torch.Tensor, return_pred = False) -> torch.Tensor:
        """
        Compute the loss of the MLP
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_action: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        y_pred = self.forward(latent)
        if return_pred:
            return self.loss_fn(y_pred, gt_action) , y_pred
        else:
            return self.loss_fn(y_pred, gt_action)

class GMMHead(PredHead): 
    """
    A 2 layer MLP that outputs the mean and log_std of a GMM distribution
    Reference: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/models/policy_nets.py#L397
    Args:
        input_dim: int, input dimension
        output_dim: int, output dimension
        num_components: int, number of components in the GMM
        activation: nn.Module, activation function
    """
    def __init__(
        self,
        input_dim : int,
        hidden_features : int,
        output_dim : int,
        num_components : int = 3,
        min_std : float = 0.01,
        activation : nn.Module = nn.ReLU,
        num_layers=2,
        fixed_scale: bool = True,
        std_activation: str = "softplus",
        use_tanh: bool = True,
    ):
        super(GMMHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_features = hidden_features
        self.output_dim = output_dim
        self.num_components = num_components

        if num_layers > 0:
            sizes = [input_dim] + [hidden_features] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.min_std = min_std

        # generate three heads
        self.mean_fc = nn.Linear(hidden_features, output_dim * num_components)
        self.logits_fc = nn.Linear(hidden_features, num_components)

        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = nn.Parameter(torch.zeros(1, num_components, output_dim), requires_grad=True)
        else:
            self.scale_fc = nn.Linear(hidden_features, output_dim * num_components)

        if std_activation == "softplus":
            self.std_actv = F.softplus
        else:
            self.std_actv = torch.exp

        self.use_tanh = use_tanh

    def _forward_model(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate the mean, scale and logits of the GMM distribution
        Args: 
            x: torch.Tensor, (B, input_dim)
        Returns:
            Dict[str, torch.Tensor], dictionary containing mean, scale and logits
                mean: torch.Tensor, (B, num_components, output_dim)
                scale: torch.Tensor, (B, num_components, output_dim)
                logits: torch.Tensor, (B, num_components)
        """
        x = self.share(x)
        mean = self.mean_fc(x).reshape(-1, self.num_components, self.output_dim)
        if self.fixed_scale:
            scale = self.scale.repeat(x.size(0), 1, 1)
        else:
            scale = self.scale_fc(x).reshape(-1, self.num_components, self.output_dim)
        logits = self.logits_fc(x)
        return {"mean": mean, "scale": scale, "logits": logits}

    def forward(self, x : torch.Tensor) -> D.Distribution:
        """
        Generate the GMM distribution
        Args: 
            x: torch.Tensor, (B, input_dim)
        Returns:
            D.Distribution, GMM distribution
        """
        gmm_stats = self._forward_model(x)
        mean, scale, logits = gmm_stats["mean"], gmm_stats["scale"], gmm_stats["logits"]

        if self.use_tanh:
            mean = torch.tanh(mean)

        # generate gmm distribution
        if self.training:
            scales = self.std_actv(scale) + self.min_std
        else:
            scales = torch.ones_like(mean) * 1e-4 

        component_distribution = D.Normal(loc=mean, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)
        
        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        return dist
    
    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Sample from the GMM distribution
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        dist = self.forward(x)
        action = dist.sample()
        return action


    def loss(self, latent : torch.Tensor, gt_action : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the GMM distribution
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_action: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        dist = self.forward(latent)
        log_prob = dist.log_prob(gt_action)
        loss = -log_prob
        return loss
    
class DiffusionHead(PredHead): 
    """Using a diffusion model as the prediction head
    
    Args: 
        input_dim: int, input dimension
        hidden_features: int, hidden dimension
        output_dim: int, output dimension
        noise_scheduler: DDIMScheduler, noise scheduler
        inference_steps: int, number of inference steps
    """
    def __init__(
        self, 
        input_dim : int, 
        hidden_features : int, 
        output_dim : int, 
        train_noise_scheduler : DDPMScheduler,
        inference_noise_scheduler : DDIMScheduler,
        inference_steps=None,
    ):
        super(DiffusionHead, self).__init__()
        self.model = ConditionedDiffusion(
            output_dim=output_dim, 
            hidden_dim=hidden_features, 
            cond_dim=input_dim, 
            train_noise_scheduler=train_noise_scheduler,
            inference_noise_scheduler=inference_noise_scheduler,
            inference_steps=inference_steps
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.model.conditional_sample(x)
    
    def pred(self, x : torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the diffusion model
        Args:
            x: torch.Tensor, (B, input_dim)
        Returns:
            torch.Tensor, (B, output_dim)
        """
        return self.forward(x)
    
    def loss(self, latent : torch.Tensor, gt_action : torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the diffusion model
        Args:
            latent: torch.Tensor, (B, input_dim)
            gt_action: torch.Tensor, (B, output_dim)
        Returns:
            torch.Tensor, scalar
        """
        return self.model.calculate_loss(latent, gt_action)