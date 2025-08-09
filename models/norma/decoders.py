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

    def forward(self, x):  # x: (B, D)
        pi = F.softmax(self.pi(x), dim=-1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return pi, mu, log_var

class QuantileRegressionHead(nn.Module):
    """Quantile Regression Head for uncertainty estimation"""
    def __init__(self, d_model, num_quantiles=9):
        super().__init__()
        self.num_quantiles = num_quantiles
        # Default quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.quantile_predictors = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_quantiles)
        ])
    
    def forward(self, x):  # x: (B, D)
        quantile_predictions = []
        for predictor in self.quantile_predictors:
            pred = predictor(x)  # (B, 1)
            quantile_predictions.append(pred)
        return torch.cat(quantile_predictions, dim=-1)  # (B, num_quantiles)

class DiffusionHead(nn.Module):
    """Denoising Diffusion Model Head"""
    def __init__(self, d_model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model + 1, d_model),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, timestep=None):  # x: (B, D), timestep: (B,) or None
        if timestep is None:
            # During inference, we'll need to handle this differently
            return self.noise_predictor(x)
        
        # Embed timestep
        timestep_emb = self.timestep_embed(timestep.unsqueeze(-1).float())  # (B, 1)
        
        # Concatenate with input
        x_with_time = torch.cat([x, timestep_emb], dim=-1)  # (B, D+1)
        
        # Predict noise
        noise_pred = self.noise_predictor(x_with_time)  # (B, 1)
        return noise_pred

class NLLHead(nn.Module):
    """Negative Log Likelihood Head for uncertainty estimation"""
    def __init__(self, d_model):
        super().__init__()
        self.mean_predictor = nn.Linear(d_model, 1)
        self.log_var_predictor = nn.Linear(d_model, 1)
    
    def forward(self, x):  # x: (B, D)
        mean = self.mean_predictor(x)  # (B, 1)
        log_var = self.log_var_predictor(x)  # (B, 1)
        return torch.cat([mean, log_var], dim=-1)  # (B, 2)

class StandardHead(nn.Module):
    """Standard regression head"""
    def __init__(self, d_model):
        super().__init__()
        self.predictor = nn.Linear(d_model, 1)
    
    def forward(self, x):  # x: (B, D)
        return self.predictor(x)  # (B, 1)

class DecoderFactory:
    """Factory class to create different decoder heads"""
    
    @staticmethod
    def create_decoder(decoder_type, d_model, **kwargs):
        """Create a decoder head based on type"""
        if decoder_type == 'mdn':
            num_components = kwargs.get('num_components', 1)
            return MDNHead(d_model, num_components)
        
        elif decoder_type == 'quantile':
            num_quantiles = kwargs.get('num_quantiles', 9)
            return QuantileRegressionHead(d_model, num_quantiles)
        
        elif decoder_type == 'diffusion':
            num_timesteps = kwargs.get('num_timesteps', 1000)
            beta_start = kwargs.get('beta_start', 1e-4)
            beta_end = kwargs.get('beta_end', 0.02)
            return DiffusionHead(d_model, num_timesteps, beta_start, beta_end)
        
        elif decoder_type == 'nll':
            return NLLHead(d_model)
        
        elif decoder_type == 'standard':
            return StandardHead(d_model)
        
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

# Loss functions for different decoders
class MDNLoss(nn.Module):
    """Loss function for Mixture Density Network"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pi, mu, log_var, target):
        # pi: (B, num_components), mu: (B, num_components), log_var: (B, num_components)
        # target: (B,)
        
        # Expand target for broadcasting
        target = target.unsqueeze(-1).expand(-1, pi.shape[1])  # (B, num_components)
        
        # Compute log probability for each component
        var = torch.exp(log_var)
        log_prob = -0.5 * (log_var + (target - mu) ** 2 / var)
        
        # Weight by mixture weights and sum
        weighted_log_prob = log_prob + torch.log(pi + 1e-8)
        
        # Log-sum-exp trick for numerical stability
        max_log_prob = torch.max(weighted_log_prob, dim=1, keepdim=True)[0]
        log_sum_exp = max_log_prob + torch.log(torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=1, keepdim=True))
        
        return -torch.mean(log_sum_exp)

class QuantileLoss(nn.Module):
    """Loss function for Quantile Regression"""
    def __init__(self, quantiles=None):
        super().__init__()
        if quantiles is None:
            self.quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        else:
            self.quantiles = quantiles
    
    def forward(self, predictions, target):
        # predictions: (B, num_quantiles), target: (B,)
        target = target.unsqueeze(-1).expand(-1, predictions.shape[1])  # (B, num_quantiles)
        
        # Quantile loss
        diff = target - predictions
        loss = torch.mean(torch.where(diff >= 0, 
                                    self.quantiles * diff, 
                                    (self.quantiles - 1) * diff))
        return loss

class DiffusionLoss(nn.Module):
    """Loss function for Diffusion Model"""
    def __init__(self):
        super().__init__()
    
    def forward(self, noise_pred, noise_target):
        # Simple MSE loss for noise prediction
        return F.mse_loss(noise_pred, noise_target)

def get_loss_fn(decoder_type, **kwargs):
    """Get appropriate loss function for decoder type"""
    if decoder_type == 'mdn':
        return MDNLoss()
    
    elif decoder_type == 'quantile':
        quantiles = kwargs.get('quantiles', None)
        return QuantileLoss(quantiles)
    
    elif decoder_type == 'diffusion':
        return DiffusionLoss()
    
    elif decoder_type == 'nll':
        def nll_loss(output, target):
            mean, log_var = output[:, 0], output[:, 1]
            var = torch.exp(log_var)
            return torch.mean(0.5 * (log_var + (target - mean) ** 2 / var))
        return nll_loss
    
    elif decoder_type == 'standard':
        return nn.MSELoss()
    
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}") 