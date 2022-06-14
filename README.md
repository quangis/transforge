# transformation-algebra

[![](https://img.shields.io/pypi/v/transformation-algebra)](https://pypi.org/project/transformation-algebra/)

A transformation language is used to describe *processes*, like those 
underlying workflows or tools, as abstract *transformations between types*. 
These types do not necessarily denote any concrete data structure, and an 
expression needs not be associated with any particular implementation. The goal 
is merely to capture some properties that are deemed *conceptually* relevant, 
in order to produce annotations of *procedural metadata*.

1.  This package facilitates defining a transformation language and parsing 
    expressions of said language.

2.  In order to reason about about transformations, we implemented a [type 
    inference](https://en.wikipedia.org/wiki/Type_inference) module in Python. 
    It accommodates both [subtype-](https://en.wikipedia.org/wiki/Subtyping) 
    and [parametric](https://en.wikipedia.org/wiki/Parametric_polymorphism) 
    polymorphism.

3.  Finally, transformation expressions can be combined and serialized into 
    [RDF](https://en.wikipedia.org/wiki/Resource_Description_Framework) graphs. 
    These transformation graphs can themselves be turned into 
    [SPARQL](https://en.wikipedia.org/wiki/SPARQL) queries and matched against 
    other transformation graphs. This enables flexible searching through 
    processes.

The library was developed for the core concept transformation algebra for 
geographical information ([CCT](https://github.com/quangis/cct)), which may act 
as an example. The [doc/](doc/) directory contains a cursory user's guide.

**This library is still in development. Interfaces might change without 
warning.**
