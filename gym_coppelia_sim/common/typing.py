from typing import Any, Dict, Tuple, Union

import numpy as np

ArrayStruct = Union[Dict[str, np.ndarray], np.ndarray]
StepInfo = Dict[str, Any]
EnvironmentTuple = Tuple[ArrayStruct, float, bool, StepInfo]
NumpyOrFloat = Union[float, np.ndarray]
