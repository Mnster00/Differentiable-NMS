import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableNMS(nn.Module):
    """
    Differentiable Non-Maximum Suppression via Hungarian Matching.
    
    This module implements the differentiable NMS approach described in the paper
    "Fabric Defect Detection via Differentiable NMS". It reformulates NMS as a bipartite
    matching problem solved through the Sinkhorn-Knopp algorithm.
    
    Args:
        alpha (float): Weight for confidence score in cost calculation. Default: 1.0
        beta (float): Weight for feature similarity in cost calculation. Default: 1.0
        gamma (float): Weight for spatial distance in cost calculation. Default: 1.0
        temperature (float): Temperature parameter for Sinkhorn normalization. Default: 0.1
        max_iter (int): Maximum number of Sinkhorn iterations. Default: 10
        eps (float): Small constant for numerical stability. Default: 1e-10
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, temperature=0.1, max_iter=10, eps=1e-10):
        super(DifferentiableNMS, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.max_iter = max_iter
        self.eps = eps
        
    def compute_cost_matrix(self, scores, features, boxes, centroids_feat=None, centroids_spatial=None):
        """
        Compute the cost matrix between proposals and latent defect regions.
        
        Args:
            scores (torch.Tensor): Confidence scores of proposals. Shape: [batch_size, num_proposals]
            features (torch.Tensor): Feature vectors of proposals. Shape: [batch_size, num_proposals, feat_dim]
            boxes (torch.Tensor): Bounding boxes of proposals. Shape: [batch_size, num_proposals, 4] (x1, y1, x2, y2)
            centroids_feat (torch.Tensor, optional): Feature centroids of latent defect regions.
                                                   Shape: [batch_size, num_regions, feat_dim]
            centroids_spatial (torch.Tensor, optional): Spatial centroids of latent defect regions.
                                                      Shape: [batch_size, num_regions, 2] (cx, cy)
                                                      
        Returns:
            torch.Tensor: Cost matrix. Shape: [batch_size, num_proposals, num_regions]
        """
        batch_size, num_proposals = scores.shape
        
        # If centroids are not provided, initialize them
        if centroids_feat is None or centroids_spatial is None:
            num_regions = max(1, int(torch.sum(scores > 0.5, dim=1).float().mean().item()))
            
            # Initialize feature centroids using k-means-like initialization
            if centroids_feat is None:
                # Use top-k proposals as initial centroids
                _, topk_indices = torch.topk(scores, min(num_regions, num_proposals), dim=1)
                batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, topk_indices.size(1))
                centroids_feat = features[batch_indices, topk_indices]
                
                # Pad if necessary
                if num_regions > topk_indices.size(1):
                    pad_size = num_regions - topk_indices.size(1)
                    pad = torch.zeros(batch_size, pad_size, features.size(2), device=features.device)
                    centroids_feat = torch.cat([centroids_feat, pad], dim=1)
            
            # Initialize spatial centroids
            if centroids_spatial is None:
                # Convert boxes to centers
                if boxes is not None:
                    centers = torch.stack([
                        (boxes[:, :, 0] + boxes[:, :, 2]) / 2,  # cx
                        (boxes[:, :, 1] + boxes[:, :, 3]) / 2   # cy
                    ], dim=2)
                    
                    # Use top-k proposal centers as initial centroids
                    _, topk_indices = torch.topk(scores, min(num_regions, num_proposals), dim=1)
                    batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, topk_indices.size(1))
                    centroids_spatial = centers[batch_indices, topk_indices]
                    
                    # Pad if necessary
                    if num_regions > topk_indices.size(1):
                        pad_size = num_regions - topk_indices.size(1)
                        pad = torch.zeros(batch_size, pad_size, 2, device=centers.device)
                        centroids_spatial = torch.cat([centroids_spatial, pad], dim=1)
                else:
                    # If boxes are not provided, use zero initialization
                    centroids_spatial = torch.zeros(batch_size, num_regions, 2, device=features.device)
        
        num_regions = centroids_feat.size(1)
        
        # Compute confidence cost: (1 - score)
        confidence_cost = 1.0 - scores.unsqueeze(2).expand(-1, -1, num_regions)
        
        # Compute feature similarity cost
        feat_cost = torch.zeros_like(confidence_cost)
        if features is not None and centroids_feat is not None:
            # Normalize features for cosine similarity
            norm_features = F.normalize(features, p=2, dim=2)
            norm_centroids = F.normalize(centroids_feat, p=2, dim=2)
            
            # Compute pairwise cosine similarity
            similarity = torch.bmm(norm_features, norm_centroids.transpose(1, 2))
            feat_cost = 1.0 - similarity  # Convert to cost (higher similarity, lower cost)
        
        # Compute spatial distance cost
        spatial_cost = torch.zeros_like(confidence_cost)
        if boxes is not None and centroids_spatial is not None:
            # Convert boxes to centers
            centers = torch.stack([
                (boxes[:, :, 0] + boxes[:, :, 2]) / 2,  # cx
                (boxes[:, :, 1] + boxes[:, :, 3]) / 2   # cy
            ], dim=2)
            
            # Compute pairwise Euclidean distance
            expanded_centers = centers.unsqueeze(2).expand(-1, -1, num_regions, 2)
            expanded_centroids = centroids_spatial.unsqueeze(1).expand(-1, num_proposals, -1, 2)
            
            # Normalize distances by image dimensions (assuming boxes are normalized)
            spatial_cost = torch.norm(expanded_centers - expanded_centroids, dim=3)
            
            # Normalize spatial cost to [0, 1] range
            if spatial_cost.numel() > 0:
                max_dist = torch.max(spatial_cost.view(batch_size, -1), dim=1)[0].view(batch_size, 1, 1)
                max_dist = torch.clamp(max_dist, min=self.eps)
                spatial_cost = spatial_cost / max_dist
        
        # Compute final cost matrix as weighted sum
        cost_matrix = (
            self.alpha * confidence_cost + 
            self.beta * feat_cost + 
            self.gamma * spatial_cost
        )
        
        return cost_matrix, centroids_feat, centroids_spatial
    
    def sinkhorn_normalization(self, cost_matrix):
        """
        Apply Sinkhorn normalization to get a differentiable soft assignment matrix.
        
        Args:
            cost_matrix (torch.Tensor): Cost matrix. Shape: [batch_size, num_proposals, num_regions]
            
        Returns:
            torch.Tensor: Soft assignment matrix. Shape: [batch_size, num_proposals, num_regions]
        """
        batch_size, num_proposals, num_regions = cost_matrix.shape
        
        # Apply temperature scaling and convert to similarity
        similarity = torch.exp(-cost_matrix / self.temperature)
        
        # Initialize row and column scaling factors
        row_scaling = torch.ones(batch_size, num_proposals, 1, device=cost_matrix.device)
        col_scaling = torch.ones(batch_size, 1, num_regions, device=cost_matrix.device)
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            # Normalize rows
            row_sum = torch.sum(similarity * col_scaling, dim=2, keepdim=True) + self.eps
            row_scaling = 1.0 / row_sum
            
            # Normalize columns
            col_sum = torch.sum(similarity * row_scaling, dim=1, keepdim=True) + self.eps
            col_scaling = 1.0 / col_sum
        
        # Compute final soft assignment matrix
        soft_assignment = similarity * row_scaling * col_scaling
        
        return soft_assignment
    
    def aggregate_proposals(self, soft_assignment, scores, boxes, features, masks=None):
        """
        Aggregate proposals based on soft assignment to latent defect regions.
        
        Args:
            soft_assignment (torch.Tensor): Soft assignment matrix. Shape: [batch_size, num_proposals, num_regions]
            scores (torch.Tensor): Confidence scores of proposals. Shape: [batch_size, num_proposals]
            boxes (torch.Tensor): Bounding boxes of proposals. Shape: [batch_size, num_proposals, 4]
            features (torch.Tensor): Feature vectors of proposals. Shape: [batch_size, num_proposals, feat_dim]
            masks (torch.Tensor, optional): Segmentation masks of proposals. 
                                          Shape: [batch_size, num_proposals, height, width]
                                          
        Returns:
            tuple: Tuple containing aggregated scores, boxes, features, and masks (if provided)
        """
        batch_size, num_proposals, num_regions = soft_assignment.shape
        
        # Aggregate scores (weighted average)
        agg_scores = torch.bmm(scores.unsqueeze(1), soft_assignment).squeeze(1)
        
        # Aggregate boxes (weighted average)
        agg_boxes = None
        if boxes is not None:
            # Expand soft assignment for broadcasting
            expanded_assignment = soft_assignment.unsqueeze(3)  # [B, N, K, 1]
            expanded_boxes = boxes.unsqueeze(2).expand(-1, -1, num_regions, -1)  # [B, N, K, 4]
            
            # Weighted average of boxes
            agg_boxes = torch.sum(expanded_assignment * expanded_boxes, dim=1)  # [B, K, 4]
        
        # Aggregate features (weighted average)
        agg_features = None
        if features is not None:
            # Expand soft assignment for broadcasting
            expanded_assignment = soft_assignment.unsqueeze(3)  # [B, N, K, 1]
            expanded_features = features.unsqueeze(2).expand(-1, -1, num_regions, -1)  # [B, N, K, D]
            
            # Weighted average of features
            agg_features = torch.sum(expanded_assignment * expanded_features, dim=1)  # [B, K, D]
        
        # Aggregate masks (weighted average)
        agg_masks = None
        if masks is not None:
            # Reshape soft assignment for broadcasting
            s = soft_assignment.size()
            m = masks.size()
            expanded_assignment = soft_assignment.view(s[0], s[1], s[2], 1, 1)  # [B, N, K, 1, 1]
            expanded_masks = masks.unsqueeze(2).expand(-1, -1, num_regions, -1, -1)  # [B, N, K, H, W]
            
            # Weighted average of masks
            agg_masks = torch.sum(expanded_assignment * expanded_masks, dim=1)  # [B, K, H, W]
        
        return agg_scores, agg_boxes, agg_features, agg_masks
    
    def filter_low_confidence(self, scores, boxes=None, features=None, masks=None, threshold=0.05):
        """
        Filter out low confidence proposals.
        
        Args:
            scores (torch.Tensor): Confidence scores. Shape: [batch_size, num_regions]
            boxes (torch.Tensor, optional): Bounding boxes. Shape: [batch_size, num_regions, 4]
            features (torch.Tensor, optional): Feature vectors. Shape: [batch_size, num_regions, feat_dim]
            masks (torch.Tensor, optional): Segmentation masks. Shape: [batch_size, num_regions, height, width]
            threshold (float): Confidence threshold. Default: 0.05
            
        Returns:
            tuple: Tuple containing filtered scores, boxes, features, and masks
        """
        # Create mask for high confidence proposals
        keep_mask = scores > threshold  # [batch_size, num_regions]
        
        # Apply mask to scores
        filtered_scores = scores * keep_mask.float()
        
        # Apply mask to boxes if provided
        filtered_boxes = None
        if boxes is not None:
            filtered_boxes = boxes.clone()
            for b in range(boxes.size(0)):
                filtered_boxes[b, ~keep_mask[b]] = 0.0
        
        # Apply mask to features if provided
        filtered_features = None
        if features is not None:
            filtered_features = features.clone()
            for b in range(features.size(0)):
                filtered_features[b, ~keep_mask[b]] = 0.0
        
        # Apply mask to masks if provided
        filtered_masks = None
        if masks is not None:
            filtered_masks = masks.clone()
            for b in range(masks.size(0)):
                filtered_masks[b, ~keep_mask[b]] = 0.0
        
        return filtered_scores, filtered_boxes, filtered_features, filtered_masks
    
    def forward(self, scores, boxes=None, features=None, masks=None, centroids_feat=None, centroids_spatial=None):
        """
        Forward pass of the Differentiable NMS module.
        
        Args:
            scores (torch.Tensor): Confidence scores of proposals. Shape: [batch_size, num_proposals]
            boxes (torch.Tensor, optional): Bounding boxes of proposals. Shape: [batch_size, num_proposals, 4]
            features (torch.Tensor, optional): Feature vectors of proposals. Shape: [batch_size, num_proposals, feat_dim]
            masks (torch.Tensor, optional): Segmentation masks of proposals. Shape: [batch_size, num_proposals, height, width]
            centroids_feat (torch.Tensor, optional): Feature centroids of latent defect regions.
                                                   Shape: [batch_size, num_regions, feat_dim]
            centroids_spatial (torch.Tensor, optional): Spatial centroids of latent defect regions.
                                                      Shape: [batch_size, num_regions, 2]
                                                      
        Returns:
            tuple: Tuple containing filtered aggregated scores, boxes, features, and masks
        """
        # Compute cost matrix
        cost_matrix, centroids_feat, centroids_spatial = self.compute_cost_matrix(
            scores, features, boxes, centroids_feat, centroids_spatial
        )
        
        # Apply Sinkhorn normalization to get soft assignment
        soft_assignment = self.sinkhorn_normalization(cost_matrix)
        
        # Aggregate proposals based on soft assignment
        agg_scores, agg_boxes, agg_features, agg_masks = self.aggregate_proposals(
            soft_assignment, scores, boxes, features, masks
        )
        
        # Filter out low confidence proposals
        filtered_scores, filtered_boxes, filtered_features, filtered_masks = self.filter_low_confidence(
            agg_scores, agg_boxes, agg_features, agg_masks
        )
        
        return filtered_scores, filtered_boxes, filtered_features, filtered_masks, soft_assignment


# Example usage
if __name__ == "__main__":
    # Create random test data
    batch_size = 2
    num_proposals = 100
    num_regions = 10
    feat_dim = 256
    
    scores = torch.rand(batch_size, num_proposals)
    boxes = torch.rand(batch_size, num_proposals, 4)
    features = torch.rand(batch_size, num_proposals, feat_dim)
    
    # Create DNMS module
    dnms = DifferentiableNMS(alpha=1.0, beta=1.0, gamma=1.0, temperature=0.1, max_iter=10)
    
    # Apply DNMS
    filtered_scores, filtered_boxes, filtered_features, filtered_masks, soft_assignment = dnms(
        scores, boxes, features
    )
    
    print(f"Input scores shape: {scores.shape}")
    print(f"Input boxes shape: {boxes.shape}")
    print(f"Input features shape: {features.shape}")
    print(f"Output scores shape: {filtered_scores.shape}")
    print(f"Output boxes shape: {filtered_boxes.shape}")
    print(f"Output features shape: {filtered_features.shape}")
    print(f"Soft assignment shape: {soft_assignment.shape}")


# Integration example with a detection model
class DetectionModelWithDNMS(nn.Module):
    def __init__(self, backbone, rpn, roi_head, dnms_kwargs=None):
        super(DetectionModelWithDNMS, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head
        
        # Initialize DNMS module
        dnms_kwargs = dnms_kwargs or {}
        self.dnms = DifferentiableNMS(**dnms_kwargs)
    
    def forward(self, images):
        # Extract features
        features = self.backbone(images)
        
        # Generate proposals
        proposals, proposal_scores = self.rpn(features)
        
        # Extract proposal features
        proposal_features = self.roi_head.extract_features(features, proposals)
        
        # Apply DNMS
        filtered_scores, filtered_boxes, filtered_features, _, _ = self.dnms(
            proposal_scores, proposals, proposal_features
        )
        
        # Final classification and regression
        class_scores, box_deltas = self.roi_head.predict(filtered_features)
        
        return class_scores, filtered_boxes + box_deltas
