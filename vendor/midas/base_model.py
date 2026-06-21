import torch

from device_utils import get_device


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=get_device())

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
