import flax
import collections
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
InfoDict = Dict[str, Any]
