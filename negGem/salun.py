import torch
def apply_salun(gradient, threshold, mask=None):
    '''
    Applies "SalUn-like" filtering to a raw gradient tensor by keeping only the largest 
    fraction of gradient values (by absolute value) and setting the others to zero.
    
    For example, threshold=0.1 means keep only the top 10% largest gradient magnitudes.
    '''
    # Flatten the gradient for easier manipulation
    grad = gradient.view(-1)
    grad_abs = torch.abs(grad)
    
    if mask is not None:
        # If mask is a tuple (gradient, mask) from previous call, extract the mask
        if isinstance(mask, tuple) and len(mask) == 2:
            mask = mask[1]
        # Apply the mask to the gradient as well
        grad = grad.mul_(mask)
        return grad.view(gradient.shape), mask
    
    mask = grad_abs > 0.0
    
    if threshold <= 0.0:
        grad.zero_()
    elif threshold >= 1.0:
        # No filtering, keep all gradients
        return gradient
    else:
        # Find the cutoff for the top threshold fraction
        cutoff = torch.quantile(grad_abs, 1 - threshold)
        # Zero out gradients smaller than the cutoff
        mask = grad_abs >= cutoff
        grad.mul_(mask)
    
    return grad.view(gradient.shape), mask

"""
def apply_salun(gradient, threshold):
    '''
    Applies "SalUn-like" filtering to a raw gradient tensor by keeping only gradient values
    with absolute value greater than or equal to the threshold and setting others to zero.

    For example, threshold=0.01 means keep only gradient values with magnitude >= 0.01.
    '''
    # Flatten the gradient for easier manipulation
    grad = gradient.view(-1)
    grad_abs = torch.abs(grad)

    # Zero out gradients smaller than the threshold
    mask = grad_abs >= threshold
    filtered_grad = grad * mask

    return filtered_grad.view(gradient.shape)
"""
