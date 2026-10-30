from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

DataCube = NDArray[np.float32]

@dataclass(frozen=True)
class ExecutionContext:
    backend: Literal["naive", "dask"] = "naive"
    workers: int = 1
    chunk_height: int = 1

class LineOperation(Protocol):
    def __call__(self, line: DataCube) -> None:
        """Modify a matrix in the shape of (width, band), in place."""
        ...
