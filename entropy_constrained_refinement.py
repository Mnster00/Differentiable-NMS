import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EntropyConstrainedRefinement(nn.Module):
    """
    Entropy-Constrained Mask Refinement via Frank-Wolfe Optimization.
    
    This module implements the entropy-constrained mask refinement approach described
    in the paper "Fabric Defect Detection via Differentiable NMS". It optimizes proposal
    probabilities while maintaining sufficient uncertainty through entropy constraints.
    
    Args:
        entropy_threshold (float): Minimum required entropy threshold. Default: 0.6
        max_iterations (int): Maximum number of Frank-Wolfe iterations. Default: 50
        convergence_tol (float): Convergence tolerance for early stopping. Default: 1e-4
        binary_search_steps (int): Number of binary search steps for dual problem. Default: 20
        lambda_min (float): Minimum value for lambda in binary search. Default: 0.001
        lambda_max (float): Maximum value for lambda in binary search. Default: 100.0
    """
    
    def __init__(
        self,
        entropy_threshold=0.6,
        max_iterations=50,
        convergence_tol=1e-4,
        binary_search_steps=20,
        lambda_min=0.001,
        lambda_max=100.0,
    ):
        super(EntropyConstrainedRefinement, self).__init__()
        self.entropy_threshold = entropy_threshold
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.binary_search_steps = binary_search_steps
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
    
    def compute_entropy(self, p):
        """
        Compute entropy of a probability distribution.
        
        Args:
            p (torch.Tensor): Probability distribution. Shape: [batch_size, num_elements]
            
        Returns:
            torch.Tensor: Entropy values. Shape: [batch_size]
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_safe = p + eps
        p_safe = p_safe / torch.sum(p_safe, dim=1, keepdim=True)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(p_safe * torch.log(p_safe), dim=1)
        
        return entropy
    
    def solve_dual_problem(self, quality_scores, batch_idx):
        """
        Solve the dual problem for a given batch index using binary search on lambda.
        
        Args:
            quality_scores (torch.Tensor): Quality scores. Shape: [batch_size, num_elements]
            batch_idx (int): Batch index
            
        Returns:
            torch.Tensor: Optimal distribution. Shape: [num_elements]
        """
        q = quality_scores[batch_idx]
        
        # Initialize lambda values for binary search
        lambda_min = self.lambda_min
        lambda_max = self.lambda_max
        
        # Binary search to find lambda that satisfies entropy constraint
        for _ in range(self.binary_search_steps):
            lambda_mid = (lambda_min + lambda_max) / 2.0
            
            # Compute distribution for current lambda
            logits = q / lambda_mid
            s = F.softmax(logits, dim=0)
            
            # Compute entropy
            entropy = self.compute_entropy(s.unsqueeze(0)).item()
            
            # Update lambda bounds
            if entropy < self.entropy_threshold:
                lambda_min = lambda_mid
            else:
                lambda_max = lambda_mid
        
        # Compute final distribution using found lambda
        logits = q / lambda_max
        s = F.softmax(logits, dim=0)
        
        return s
    
    def line_search(self, p_t, s, quality_scores, batch_idx):
        """
        Perform line search to find optimal step size.
        
        Args:
            p_t (torch.Tensor): Current distribution. Shape: [num_elements]
            s (torch.Tensor): Direction distribution. Shape: [num_elements]
            quality_scores (torch.Tensor): Quality scores. Shape: [batch_size, num_elements]
            batch_idx (int): Batch index
            
        Returns:
            float: Optimal step size
        """
        q = quality_scores[batch_idx]
        
        # Define objective function: sum(p * q)
        def objective(gamma):
            p_new = (1 - gamma) * p_t + gamma * s
            return torch.sum(p_new * q).item()
        
        # Grid search for initial approximation
        gammas = torch.linspace(0, 1, 11, device=p_t.device)
        objectives = [objective(gamma.item()) for gamma in gammas]
        best_idx = np.argmax(objectives)
        
        # Refine with binary search around best gamma
        if best_idx == 0:
            return 0.0
        elif best_idx == len(gammas) - 1:
            return 1.0
        else:
            gamma_min = gammas[best_idx - 1].item()
            gamma_max = gammas[best_idx + 1].item()
            
            for _ in range(5):  # 5 refinement steps
                gamma_mid = (gamma_min + gamma_max) / 2.0
                obj_min = objective(gamma_min)
                obj_max = objective(gamma_max)
                obj_mid = objective(gamma_mid)
                
                if obj_mid > max(obj_min, obj_max):
                    # Mid point is best, narrow search
                    gamma_new_min = (gamma_min + gamma_mid) / 2.0
                    gamma_new_max = (gamma_mid + gamma_max) / 2.0
                    gamma_min, gamma_max = gamma_new_min, gamma_new_max
                elif obj_min > obj_max:
                    # Min side is better
                    gamma_max = gamma_mid
                else:
                    # Max side is better
                    gamma_min = gamma_mid
            
            return (gamma_min + gamma_max) / 2.0
    
    def frank_wolfe_optimization(self, quality_scores):
        """
        Apply Frank-Wolfe optimization to find optimal distribution under entropy constraint.
        
        Args:
            quality_scores (torch.Tensor): Quality scores. Shape: [batch_size, num_elements]
            
        Returns:
            torch.Tensor: Optimized probability distribution. Shape: [batch_size, num_elements]
        """
        batch_size, num_elements = quality_scores.shape
        
        # Initialize with uniform distribution
        p = torch.ones(batch_size, num_elements, device=quality_scores.device) / num_elements
        
        # Apply Frank-Wolfe optimization for each batch
        for batch_idx in range(batch_size):
            p_t = p[batch_idx]
            q = quality_scores[batch_idx]
            
            for t in range(self.max_iterations):
                # Solve linearized problem (find direction)
                s = self.solve_dual_problem(quality_scores, batch_idx)
                
                # Line search for optimal step size
                gamma_t = self.line_search(p_t, s, quality_scores, batch_idx)
                
                # Update solution
                p_new = (1 - gamma_t) * p_t + gamma_t * s
                
                # Check convergence
                improvement = torch.sum((p_new - p_t) * q).item()
                if abs(improvement) < self.convergence_tol:
                    break
                
                p_t = p_new
            
            # Update batch result
            p[batch_idx] = p_t
        
        return p
    
    def apply_spatial_coherence(self, p, masks, lambda_reg=0.1):
        """
        Apply spatial coherence regularization to refined probabilities.
        
        Args:
            p (torch.Tensor): Probability distribution. Shape: [batch_size, num_elements]
            masks (torch.Tensor): Segmentation masks. Shape: [batch_size, num_elements, height, width]
            lambda_reg (float): Regularization strength. Default: 0.1
            
        Returns:
            torch.Tensor: Regularized probability distribution. Shape: [batch_size, num_elements]
        """
        if masks is None:
            return p
        
        batch_size, num_elements = p.shape
        
        # Compute total variation for each mask
        tv_reg = torch.zeros_like(p)
        
        for b in range(batch_size):
            for k in range(num_elements):
                mask = masks[b, k]
                
                # Compute horizontal and vertical gradients
                h_grad = torch.abs(mask[:, 1:] - mask[:, :-1]).sum()
                v_grad = torch.abs(mask[1:, :] - mask[:-1, :]).sum()
                
                # Total variation is sum of gradients
                tv = h_grad + v_grad
                
                # Normalize by mask size
                tv = tv / (mask.numel() + 1e-10)
                
                # Store total variation
                tv_reg[b, k] = tv
        
        # Normalize TV values
        tv_max = torch.max(tv_reg, dim=1, keepdim=True)[0]
        tv_min = torch.min(tv_reg, dim=1, keepdim=True)[0]
        tv_range = tv_max - tv_min + 1e-10
        tv_norm = (tv_reg - tv_min) / tv_range
        
        # Apply TV regularization (lower TV is better, so we invert)
        tv_weight = 1.0 - tv_norm
        
        # Combine with current probabilities
        p_reg = (1.0 - lambda_reg) * p + lambda_reg * tv_weight
        
        # Renormalize
        p_reg = p_reg / torch.sum(p_reg, dim=1, keepdim=True)
        
        return p_reg
    
    def forward(self, scores, quality_scores=None, masks=None):
        """
        Forward pass of the Entropy-Constrained Refinement module.
        
        Args:
            scores (torch.Tensor): Initial confidence scores. Shape: [batch_size, num_elements]
            quality_scores (torch.Tensor, optional): Quality scores for optimization.
                                                  Shape: [batch_size, num_elements]
            masks (torch.Tensor, optional): Segmentation masks for spatial coherence.
                                         Shape: [batch_size, num_elements, height, width]
                                         
        Returns:
            torch.Tensor: Refined probability distribution. Shape: [batch_size, num_elements]
        """
        # If quality scores are not provided, use initial scores
        if quality_scores is None:
            quality_scores = scores.clone()
        
        # Normalize scores to valid probability distribution
        p_init = F.softmax(scores, dim=1)
        
        # Apply Frank-Wolfe optimization
        p_refined = self.frank_wolfe_optimization(quality_scores)
        
        # Apply spatial coherence regularization if masks are provided
        if masks is not None:
            p_refined = self.apply_spatial_coherence(p_refined, masks)
        
        return p_refined


# Example usage
if __name__ == "__main__":
    # Create random test data
    batch_size = 2
    num_elements = 10
    
    scores = torch.rand(batch_size, num_elements)
    quality_scores = torch.rand(batch_size, num_elements)  # In practice, this would be IoU with ground truth
    
    # Create refinement module
    refinement = EntropyConstrainedRefinement(
        entropy_threshold=0.6,
        max_iterations=50,
        convergence_tol=1e-4
    )
    
    # Apply refinement
    refined_probs = refinement(scores, quality_scores)
    
    print(f"Input scores shape: {scores.shape}")
    print(f"Input quality scores shape: {quality_scores.shape}")
    print(f"Output refined probabilities shape: {refined_probs.shape}")
    
    # Check entropy constraint
    entropies = refinement.compute_entropy(refined_probs)
    print(f"Entropies: {entropies}")
    print(f"Entropy threshold: {refinement.entropy_threshold}")
    print(f"Constraint satisfied: {(entropies >= refinement.entropy_threshold).all().item()}")


# Integration with DNMS
class DNMSWithRefinement(nn.Module):
    def __init__(self, dnms_kwargs=None, refinement_kwargs=None):
        super(DNMSWithRefinement, self).__init__()
        
        # Initialize DNMS module
        dnms_kwargs = dnms_kwargs or {}
        self.dnms = DifferentiableNMS(**dnms_kwargs)
        
        # Initialize refinement module
        refinement_kwargs = refinement_kwargs or {}
        self.refinement = EntropyConstrainedRefinement(**refinement_kwargs)
    
    def forward(self, scores, boxes=None, features=None, masks=None, quality_scores=None):
        # Apply DNMS
        filtered_scores, filtered_boxes, filtered_features, filtered_masks, soft_assignment = self.dnms(
            scores, boxes, features, masks
        )
        
        # Apply entropy-constrained refinement
        refined_probs = self.refinement(filtered_scores, quality_scores, filtered_masks)
        
        return refined_probs, filtered_boxes, filtered_features, filtered_masks, soft_assignment
