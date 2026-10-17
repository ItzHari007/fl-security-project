import torch

def sign_flip(delta):
    """Reverse the direction of the update."""
    return -delta

def random_update(delta):
    """Replace the update with Gaussian noise that has the same expected L2 norm."""
    std = delta.norm() / (delta.numel() ** 0.5)
    return torch.randn_like(delta) * std

def scaling(delta, factor=10.0):
    """Multiply the update by a constant factor."""
    return delta * factor

def apply_attack(delta, attack, scale_factor=10.0):
    if attack == "none":
        return delta
    if attack == "sign_flip":
        return sign_flip(delta)
    if attack == "random":
        return random_update(delta)
    if attack == "scaling":
        return scaling(delta, scale_factor)
    raise ValueError(f"Unknown attack: {attack}")
