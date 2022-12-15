from typing import Dict, Tuple, Union

import numpy as np

ArrayStruct = Union[Dict[str, np.ndarray], np.ndarray]
EnvironmentTuple = Tuple[ArrayStruct, float, bool, dict]
NumpyOrFloat = Union[float, np.ndarray]
