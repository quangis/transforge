# cf. <https://rdflib.readthedocs.io/en/stable/_modules/rdflib/collection.html>

from __future__ import annotations

from rdflib import Graph, RDF
from rdflib.term import BNode, Node
from typing import Iterable, Iterator


class GraphList(Graph):
    """
    An RDF graph augmented with methods for creating, reading and destroying 
    lists.
    """

    def __init__(self) -> None:
        super().__init__()

    def add_list(self, items: Iterable[Node]) -> Node:
        if not isinstance(items, (list, slice)):
            items = list(items)
        if not items:
            return RDF.nil
        node = BNode()
        self.add((node, RDF.first, items[0]))
        self.add((node, RDF.rest, self.add_list(items[1:])))
        return node

    def get_list(self, list_node: Node) -> Iterator[Node]:
        node: Node | None = list_node
        while first := self.value(node, RDF.first, any=False):
            yield first
            node = self.value(node, RDF.rest, any=False)
        if not node == RDF.nil:
            raise RuntimeError("Node is not an RDF list")

    def remove_list(self, list_node: Node) -> None:
        next_node = self.value(list_node, RDF.rest, any=False)
        if next_node:
            self.remove_list(next_node)
        self.remove((list_node, RDF.first, None))
        self.remove((list_node, RDF.rest, None))
