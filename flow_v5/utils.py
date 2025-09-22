def normalize_to_flow_range(x):
    """Convert from [0,1] to [-1,1] range for flow matching"""
    return 2.0 * x - 1.0


def to_visualization_range(x):
    """Convert from [-1,1] to [0,1] range for visualization"""
    return (x + 1.0) / 2.0
