import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical
import numpy as np


# -------------------------
# Standard Loss Functions
# -------------------------

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


# -------------------------
# Dual Mode Loss Functions
# -------------------------

class DualModeLoss(nn.Module):
    """
    Combined loss function for dual-mode conditional forecasting model.
    
    Implements:
    1. Forecasting Loss: -log P_c(y_true | Z) where c is the true condition
    2. Distribution Alignment Loss: KL divergence alignment with reference distribution
    3. Total Loss: L_forecast + λ * L_align
    
    Note: Reference distribution comes from clinical reference intervals (config.py),
    not from training data estimation.
    """
    
    def __init__(self, lambda_align=0.01, decoder_type='mdn'):
        super().__init__()
        self.lambda_align = lambda_align
        self.decoder_type = decoder_type
        
    def forecasting_loss(self, P_healthy, P_unhealthy, y_true, condition):
        """
        Compute forecasting loss: -log P_c(y_true | Z)
        
        Args:
            P_healthy: Distribution parameters for healthy condition
            P_unhealthy: Distribution parameters for unhealthy condition  
            y_true: (B,) true lab values
            condition: (B,) true conditions (1=normal, 0=abnormal)
            
        Returns:
            loss: Scalar forecasting loss
        """
        batch_size = y_true.shape[0]
        
        if self.decoder_type == 'mdn':
            # P_healthy/P_unhealthy are (pi_logits, mu, log_var) tuples
            pi_logits_h, mu_h, log_var_h = P_healthy
            pi_logits_u, mu_u, log_var_u = P_unhealthy
            
            # Create mixture distributions
            dist_healthy = self._create_mdn_distribution(pi_logits_h, mu_h, log_var_h)
            dist_unhealthy = self._create_mdn_distribution(pi_logits_u, mu_u, log_var_u)
            
        elif self.decoder_type == 'gaussian':
            # P_healthy/P_unhealthy are (mu, log_var) tuples
            mu_h, log_var_h = P_healthy
            mu_u, log_var_u = P_unhealthy
            
            dist_healthy = Normal(mu_h, torch.exp(0.5 * log_var_h))
            dist_unhealthy = Normal(mu_u, torch.exp(0.5 * log_var_u))
            
        else:
            raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")
        
        # Compute log probabilities
        log_prob_healthy = dist_healthy.log_prob(y_true)  # (B,)
        log_prob_unhealthy = dist_unhealthy.log_prob(y_true)  # (B,)
        
        # Select correct distribution based on true condition
        # condition: 0=normal/healthy, 1=abnormal/unhealthy
        log_probs = torch.where(condition == 0, log_prob_healthy, log_prob_unhealthy)
        
        # Return negative log-likelihood
        return -log_probs.mean()
    
    def alignment_loss(self, P_healthy, P_unhealthy, condition, ref_mu, ref_sigma):
        """
        Compute distribution alignment loss using KL divergence
        
        Args:
            P_healthy: Distribution parameters for healthy condition
            P_unhealthy: Distribution parameters for unhealthy condition
            condition: (B,) true conditions (0=normal, 1=abnormal)
            ref_mu: Reference distribution mean (from clinical reference intervals)
            ref_sigma: Reference distribution std (from clinical reference intervals)
            
        Returns:
            loss: Scalar alignment loss
        """
        # Create reference distribution from clinical reference intervals
        ref_dist = Normal(ref_mu, ref_sigma)
        
        if self.decoder_type == 'mdn':
            pi_logits_h, mu_h, log_var_h = P_healthy
            pi_logits_u, mu_u, log_var_u = P_unhealthy
            
            dist_healthy = self._create_mdn_distribution(pi_logits_h, mu_h, log_var_h)
            dist_unhealthy = self._create_mdn_distribution(pi_logits_u, mu_u, log_var_u)
            
        elif self.decoder_type == 'gaussian':
            mu_h, log_var_h = P_healthy
            mu_u, log_var_u = P_unhealthy
            
            dist_healthy = Normal(mu_h, torch.exp(0.5 * log_var_h))
            dist_unhealthy = Normal(mu_u, torch.exp(0.5 * log_var_u))
        
        # Compute KL divergences (approximate using sampling for complex distributions)
        kl_healthy_ref = self._approx_kl_divergence(dist_healthy, ref_dist)
        kl_unhealthy_ref = self._approx_kl_divergence(dist_unhealthy, ref_dist)
        
        # Alignment loss based on condition
        # Strategy: Always encourage healthy distribution to be close to reference,
        # and unhealthy distribution to be different from reference
        
        normal_mask = (condition == 0).float()  # (B,)
        abnormal_mask = (condition == 1).float()  # (B,)
        
        # Always encourage healthy distribution to align with reference
        healthy_alignment = torch.abs(kl_healthy_ref)  # Use absolute value for symmetry
        
        # Always encourage unhealthy distribution to diverge from reference
        # Use negative KL with a margin to encourage divergence
        unhealthy_divergence = torch.clamp(1.0 - kl_unhealthy_ref, min=0.0) ** 2
        
        # Weight the losses by the true conditions in the batch
        alignment_loss = healthy_alignment.mean() + unhealthy_divergence.mean()
        
        return alignment_loss
    
    def forward(self, P_healthy, P_unhealthy, y_true, condition, ref_mu, ref_sigma):
        """
        Compute total loss: L_forecast + λ * L_align
        
        Args:
            P_healthy: Distribution parameters for healthy condition
            P_unhealthy: Distribution parameters for unhealthy condition
            y_true: (B,) true lab values
            condition: (B,) true conditions (0=normal, 1=abnormal)
            ref_mu: Reference distribution mean (from clinical intervals)
            ref_sigma: Reference distribution std (from clinical intervals)
            
        Returns:
            total_loss: Scalar total loss
            forecast_loss: Scalar forecasting loss component
            align_loss: Scalar alignment loss component
        """
        # Compute individual losses
        forecast_loss = self.forecasting_loss(P_healthy, P_unhealthy, y_true, condition)
        align_loss = self.alignment_loss(P_healthy, P_unhealthy, condition, ref_mu, ref_sigma)
        
        # Combine losses
        total_loss = forecast_loss + self.lambda_align * align_loss
        
        return total_loss, forecast_loss, align_loss
    
    def _create_mdn_distribution(self, pi_logits, mu, log_var):
        """Create mixture distribution from MDN parameters"""
        # pi_logits: (B, n_components) - mixture weights
        # mu: (B, n_components) - component means  
        # log_var: (B, n_components) - component log variances
        
        categorical = Categorical(logits=pi_logits)
        components = Normal(mu, torch.exp(0.5 * log_var))
        
        return MixtureSameFamily(categorical, components)
    
    def _approx_kl_divergence(self, dist_p, dist_q, n_samples=100):
        """
        Approximate KL(p || q) using Monte Carlo sampling
        
        KL(p || q) = E_p[log p(x) - log q(x)]
        """
        # Sample from distribution p
        samples = dist_p.sample((n_samples,))  # (n_samples, B)
        
        # Compute log probabilities
        log_p = dist_p.log_prob(samples)  # (n_samples, B)
        log_q = dist_q.log_prob(samples)  # (n_samples, B)
        
        # Approximate KL divergence
        kl = (log_p - log_q).mean(dim=0)  # (B,)
        
        return kl


# -------------------------
# Single Mode Loss Functions  
# -------------------------
class SingleModeLoss(nn.Module):
    """
    Loss function for single mode conditional transformer.
    Handles different losses based on the condition (healthy vs unhealthy).
    """
    def __init__(self, lambda_align=0.01, adaptive_weight=True):
        super().__init__()
        self.lambda_align = lambda_align
        self.adaptive_weight = adaptive_weight
        self.forecast_loss_fn = nn.GaussianNLLLoss()
        
    def forward(self, mu, log_var, y_true, condition, ref_mu, ref_sigma):
        """
        Compute loss for single mode model
        
        Args:
            mu: (B,) - predicted means
            log_var: (B,) - predicted log variances
            y_true: (B,) - true values
            condition: (B,) - conditions (1=normal, 0=abnormal)
            ref_mu: (B,) - reference means
            ref_sigma: (B,) - reference standard deviations
            
        Returns:
            total_loss: Scalar total loss
            forecast_loss: Scalar forecasting loss component
            align_loss: Scalar alignment loss component
        """
        # Forecasting loss (Gaussian NLL)
        forecast_loss = self.forecast_loss_fn(mu, y_true, torch.exp(log_var))
        
        # Create predicted and reference distributions
        pred_dist = Normal(mu, torch.exp(0.5 * log_var))
        ref_dist = Normal(ref_mu, ref_sigma)
        
        kl_div = torch.distributions.kl_divergence(pred_dist, ref_dist)
        kl_weight = torch.exp(log_var) / (torch.exp(log_var) + ref_sigma**2 + 1e-6)
        if not self.adaptive_weight:
            kl_weight = torch.ones_like(kl_weight)

        # Split predictions based on condition
        normal_mask = (condition == 1)
        abnormal_mask = (condition == 0)
        
        # For normal patients: encourage alignment with reference (more when uncertain)
        normal_alignment = (
        (kl_div[normal_mask] * kl_weight[normal_mask]).mean()
        if normal_mask.any() else torch.tensor(0.0, device=mu.device)
    )

        # For abnormal patients: encourage divergence from reference (more when uncertain)
        abnormal_divergence = torch.clamp(1.0 - kl_div[abnormal_mask], min=0.0).pow(2)
        abnormal_alignment = (
            (abnormal_divergence * kl_weight[abnormal_mask]).mean()
            if abnormal_mask.any() else torch.tensor(0.0, device=mu.device)
        )
        align_loss = normal_alignment + abnormal_alignment
        
        # Total loss
        total_loss = forecast_loss + self.lambda_align * align_loss
        
        return total_loss, forecast_loss, align_loss

