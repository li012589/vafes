import torch
import torch.nn as nn


def put(obj, device=torch.device('cpu'), dtype=torch.float32):
    """
    Recursively move objects in the dictionary to a specified device or change their dtype.

    Args:
        obj: The object to process. Can be a dictionary, list, tuple, or any other type.
        device (torch.device): The target device for torch.Tensor and torch.nn.Module objects.
        dtype (torch.dtype): The target data type for torch.Tensor objects.

    Returns:
        The processed object with tensors and modules moved to the specified device or changed dtype.
    """
    if isinstance(obj, dict):
        return {key: put(value, device=device, dtype=dtype) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [put(item, device=device, dtype=dtype) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(put(item, device=device, dtype=dtype) for item in obj)
    elif isinstance(obj, (nn.Module, nn.ParameterDict, nn.ParameterList)):
        obj = obj.to(device, dtype)
        return obj
    elif isinstance(obj, (torch.Tensor, nn.Parameter)):
        requiresGrad = obj.requires_grad
        with torch.no_grad():
            obj = obj.to(device, dtype)
        if requiresGrad:
            obj.requires_grad_()
        return obj
    else:
        return obj


def yieldTrainable(data):
    # Generator function to recursively yield torch.Tensor objects from nested data
    if isinstance(data, dict):
        for value in data.values():
            yield from yieldTrainable(value)
    elif isinstance(data, (nn.Module, nn.ParameterDict, nn.ParameterList)):  # Check if it's an nn.Module
        for param in data.parameters():
            yield param
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield from yieldTrainable(item)
    elif isinstance(data, (torch.Tensor, nn.Parameter)):
        yield data
