from abc import ABC, abstractmethod
from typing import Any


class ReprID(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        """__repr__ is used for id and eq, it should be redefined in children."""
        pass

    def __hash__(self) -> int:
        """Hash assumes important parameters are included in __repr__."""
        return hash(self.__repr__())

    def __eq__(self, other: Any) -> bool:
        return self.__hash__() == other.__hash__()

    def __lt__(self, other: Any) -> bool:
        """lt is necessary for things like np.unique(). Ordering is loosely defined based on start times."""
        return self.start < other.start
