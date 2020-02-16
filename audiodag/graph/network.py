from dataclasses import dataclass
from audiodag.graph.node import Node
from typing import List, Tuple


@dataclass
class Graph:
    nodes: List[Node]


if __name__ == "__main__":

    Node(data='node_1',
         )