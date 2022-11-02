# transforge

[![](https://img.shields.io/pypi/v/transforge)](https://pypi.org/project/transforge/)

A transformation language is used to describe *processes*, like the ones 
implicit in workflows or software tools, as *transformations between 
types*. These types do not necessarily denote a concrete data structure, 
and the descriptions may be independent from any particular 
implementation. The goal is only to capture some properties that are 
deemed *conceptually* relevant, as a form of *procedural metadata*.

`transforge` facilitates defining a transformation language and parsing 
its expressions into semantic graphs. It is written in pure Python with 
few dependencies.

In order to reason about about transformations, it also implements a 
[type inference](https://en.wikipedia.org/wiki/Type_inference) module, 
which accommodates both 
[subtype-](https://en.wikipedia.org/wiki/Subtyping) and 
[parametric](https://en.wikipedia.org/wiki/Parametric_polymorphism) 
polymorphism.

Expressions of a transformation language can be turned into 
transformation graphs. Such graphs can be serialized into 
[RDF](https://en.wikipedia.org/wiki/Resource_Description_Framework), or 
converted to [SPARQL](https://en.wikipedia.org/wiki/SPARQL) queries and 
matched against other graphs. This enables flexible searching through 
processes.

**The library is still in development. Interfaces might change without 
warning.**


### Usage

This package was developed for the [CCT](https://github.com/quangis/cct) 
algebra for geographical information, which may act as an example.

Click here for a [tutorial](docs/tutorial.md).
