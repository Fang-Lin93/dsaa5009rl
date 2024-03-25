
import numpy as np
from dataset import Batch
from typing import Dict, Any
InfoDict = Dict[str, Any]


class Agent(object):
    name = 'agent'

    def update(self, batch: Batch) -> InfoDict:
        """
        How to update the model?
        """
        raise NotImplementedError

    def sample_actions(self, *args, **kwargs) -> np.ndarray:
        """
        How to take the actions
        """

        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
