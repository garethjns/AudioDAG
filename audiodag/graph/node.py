from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class Edge:
    weight: float = 1.0


@dataclass
class Node:
    data: Any
    connections_out: List[Tuple[float, "Node"]]
    terminal: bool = False

