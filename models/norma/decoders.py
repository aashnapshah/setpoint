import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MDNHead(nn.Module):
    """Mixture Density Network Head for multi-modal predictions"""
    def __init__(self, d_model, num_components=1):
        super().__init__()
        self.num_components = num_components
        self.pi = nn.Linear(d_model, num_components)
        self.mu = nn.Linear(d_model, num_components)
        self.log_var = nn.Linear(d_model, num_components)
        
        # Initialize weights for numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for numerical stability"""
        # Initialize pi logits to small values (near zero) to start with uniform mixture
        nn.init.normal_(self.pi.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pi.bias, 0.0)
        
        # Initialize mu weights normally
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.constant_(self.mu.bias, 0.0)
        
        # Initialize log_var to small negative values (small initial variance)
        nn.init.normal_(self.log_var.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.log_var.bias, -2.0)  # exp(-2) = ~0.135 initial std

    def forward(self, x):  # x: (B, D)
        pi_logits = self.pi(x)  # Return raw logits, not softmax
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        # Add clipping for numerical stability
        pi_logits = torch.clamp(pi_logits, min=-10.0, max=10.0)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        
        return pi_logits, mu, log_var

class StandardHead(nn.Module):
    """Standard regression head"""
    def __init__(self, d_model):
        super().__init__()
        self.predictor = nn.Linear(d_model, 1)
    
    def forward(self, x):  # x: (B, D)
        return self.predictor(x)  # (B, 1)

class GaussianHead(nn.Module):
    """Predicts parameters of a single 1D Gaussian (mu, log_var)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.log_var = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, D)
        mu = self.mu(x).squeeze(-1)
        log_var = self.log_var(x).squeeze(-1)
        return mu, log_var

class DecoderFactory:
    @staticmethod
    def create_decoder(decoder_type, d_model, **kwargs):
        if decoder_type == 'mdn':
            num_components = kwargs.get('num_components', 1)
            return MDNHead(d_model, num_components)
        elif decoder_type == 'standard':
            return StandardHead(d_model)
        elif decoder_type == 'gaussian':
            return GaussianHead(d_model)
        else:
            raise ValueError("Only MDN, standard, and gaussian decoder types supported")

def get_loss_fn(decoder_type, **kwargs):
    if decoder_type == 'mdn':
        return MDNLoss(anchor_weight=kwargs.get('anchor_weight', 1.0))
    elif decoder_type == 'standard':
        return nn.MSELoss()
    else:
        raise ValueError("Only MDN and standard decoder types supported for regular models")

# Loss function for MDN
class MDNLoss(nn.Module):
    """Anchored MDN loss: mixture NLL + KL anchor of component 0 to reference Normal(ref_mu, sqrt(ref_var))."""
    def __init__(self, anchor_weight: float = 1.0):
        super().__init__()
        self.anchor_weight = anchor_weight
    
    def forward(self, pi_logits, mu, log_var, target, ref_mu=None, ref_var=None):
        # pi_logits: (B, K), mu: (B, K), log_var: (B, K); target: (B,)
        # Convert logits to probabilities
        pi = F.softmax(pi_logits, dim=-1)
        
        # Mixture negative log-likelihood
        target = target.unsqueeze(-1).expand_as(pi)  # (B, K)
        var = torch.exp(log_var)
        log_prob = -0.5 * (log_var + (target - mu) ** 2 / var)  # (B, K)
        weighted_log_prob = log_prob + torch.log(pi + 1e-8)     # (B, K)
        max_log_prob = torch.max(weighted_log_prob, dim=1, keepdim=True)[0]
        log_sum_exp = max_log_prob + torch.log(torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=1, keepdim=True))
        nll = -torch.mean(log_sum_exp)

        # Optional reference anchoring for component 0
        anchor_loss = 0.0
        if (ref_mu is not None) and (ref_var is not None):
            eps = 1e-8
            ref_std = torch.sqrt(torch.clamp(ref_var, min=eps))  # (B,)
            # Take component 0 as the reference-anchored component
            mu0 = mu[:, 0]
            std0 = torch.exp(0.5 * log_var[:, 0])
            dist0 = torch.distributions.Normal(mu0, std0)
            dist_ref = torch.distributions.Normal(ref_mu, ref_std)
            kl0 = torch.distributions.kl_divergence(dist0, dist_ref)
            anchor_loss = kl0.mean()
        
        return nll + self.anchor_weight * anchor_loss