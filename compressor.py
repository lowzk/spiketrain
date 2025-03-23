import numpy as np
import torch

def pack_tensor(tensor: torch.Tensor):
    """
    Packs a tensor of 1s and 0s into a space-optimized representation.
    
    Args:
        tensor (torch.Tensor): A float32 tensor containing 1s and 0s.
    
    Returns:
        torch.Tensor: A packed tensor (torch.uint8) with 1 bit per element.
        tuple: The original shape of the tensor for unpacking.
    """
    # Ensure the tensor is a float and convert to boolean (0 -> False, 1 -> True)
    tensor = tensor.to(torch.bool)
    original_shape = tensor.shape
    
    # Flatten the tensor and convert to numpy for bit packing
    flattened = tensor.flatten().numpy().astype(np.uint8)
    packed = np.packbits(flattened)  # Packs 8 boolean values into 1 byte
    
    # Convert back to a torch tensor
    packed_tensor = torch.from_numpy(packed).to(torch.uint8)
    return packed_tensor, original_shape

def unpack_tensor(packed: torch.Tensor, original_shape: tuple):
    """
    Unpacks a packed tensor back into its original form.
    
    Args:
        packed (torch.Tensor): A packed tensor (torch.uint8) with 1 bit per element.
        original_shape (tuple): The original shape of the tensor.
    
    Returns:
        torch.Tensor: The unpacked tensor.
    """
    # Convert to numpy and unpack the bits
    unpacked = np.unpackbits(packed.numpy())
    
    # Convert back to a torch tensor and reshape
    unpacked_tensor = torch.from_numpy(unpacked).to(torch.float32)
    unpacked_tensor = unpacked_tensor[:np.prod(original_shape)].reshape(original_shape)
    return unpacked_tensor


