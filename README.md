
# DNMS-FDD: Fabric Defect Detection via Differentiable NMS


A PyTorch implementation of "End-to-End Fabric Defect Detection via Differentiable NMS".



## Features

- **Fully Differentiable**: End-to-end trainable detection pipeline without gradient flow interruption
- **Hungarian Matching NMS**: Reformulates NMS as a bipartite matching problem solved through the Sinkhorn-Knopp algorithm
- **Entropy-Constrained Refinement**: Optimizes proposal masks through principled uncertainty modeling


### Basic Usage of DNMS Module

```python
from differentiable_nms import DifferentiableNMS

# Create DNMS module
dnms = DifferentiableNMS(
    alpha=1.0,  # confidence score weight
    beta=1.0,   # feature similarity weight
    gamma=1.0,  # spatial distance weight
    temperature=0.1  # Sinkhorn temperature
)

# Apply DNMS to detection outputs
filtered_scores, filtered_boxes, filtered_features, filtered_masks, soft_assignment = dnms(
    scores, boxes, features, masks
)

## Requirements

- Python 3.6+
- PyTorch 1.9+
- torchvision 0.10+
- numpy


