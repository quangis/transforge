# transformation-algebra

[![](https://img.shields.io/pypi/v/transformation-algebra)](https://pypi.org/project/transformation-algebra/)

A transformation language is used to describe *processes*, like those that 
underlie workflows or tools, as abstract *transformations between types*. These 
types do not necessarily denote a concrete data structure, and expressions may 
be independent from any particular implementation. The goal is merely to 
capture some properties that are deemed *conceptually* relevant, in order to 
provide *procedural metadata*.

The `transformation-algebra` library facilitates defining a transformation 
language and parsing its expressions.

In order to reason about about transformations, it also implements a [type 
inference](https://en.wikipedia.org/wiki/Type_inference) module in Python, 
which accommodates both [subtype-](https://en.wikipedia.org/wiki/Subtyping) and 
[parametric](https://en.wikipedia.org/wiki/Parametric_polymorphism) 
polymorphism.

Finally, transformation expressions can be turned into transformation graphs. 
Such graphs can be serialized into 
[RDF](https://en.wikipedia.org/wiki/Resource_Description_Framework), or 
converted to [SPARQL](https://en.wikipedia.org/wiki/SPARQL) queries and matched 
against other graphs. This enables flexible searching through processes.

**The library is still in development. Interfaces might change without 
warning.**


### Usage

This package was developed for the [CCT](https://github.com/quangis/cct) 
algebra for geographical information, which may act as an example. We provide a 
cursory user's guide below.

-   [Types](01%20Types.md)
-   [Expressions](02%20Expressions.md)
-   [Querying](03%20Querying.md)
