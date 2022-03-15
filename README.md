# transformation-algebra

[![](https://img.shields.io/pypi/v/transformation-algebra)](https://pypi.org/project/transformation-algebra/)

A transformation language can describe *processes*, like workflows or tools, as 
abstract *transformations between types*. The types do not necessarily denote 
any concrete data structure, and the transformation expression is not 
associated with any particular implementation. The goal is merely to capture 
some *conceptual* properties that are deemed relevant, in order to annotate 
*procedural metadata*.

-  In order to reason about about transformations, we implemented a stand-alone 
   [type inference](https://en.wikipedia.org/wiki/Type_inference) module in 
   Python. It accommodates both 
   [subtype-](https://en.wikipedia.org/wiki/Subtyping) and 
   [parametric](https://en.wikipedia.org/wiki/Parametric_polymorphism) 
   polymorphism.

-  To enable flexible searching through processes, transformation expressions 
   can be serialized into 
   [RDF](https://en.wikipedia.org/wiki/Resource_Description_Framework) graphs 
   and queried through [SPARQL](https://en.wikipedia.org/wiki/SPARQL).

The rest of this document intended to be a cursory user's guide. The library 
was developed for the core concept transformation algebra for geographical 
information ([CCT](https://github.com/quangis/cct)) — the contents of that 
repository may act as an example.

**This library is still in development. Interfaces might change without 
warning.**

## Concrete types and subtypes

To specify type transformations, we first need to declare *base types*. To 
this end, we use the `TypeOperator` class.

    >>> from transformation_algebra import *
    >>> Real = TypeOperator(name="Real")

Base types may have supertypes. For instance, anything of type `Int` is also 
automatically of type `Real`, but not necessarily of type `Nat`:

    >>> Int = TypeOperator(name="Int", supertype=Real)
    >>> Nat = TypeOperator(name="Nat", supertype=Int)
    >>> Int <= Real
    True
    >>> Int <= Nat
    False

*Complex types* take other types as parameters. For example, `Set(Int)` could 
represent the type of sets of integers. This would automatically be a subtype 
of `Set(Real)`.

    >>> Set = TypeOperator(name="Set", params=1)
    >>> Set(Int) <= Set(Real)
    True

A special complex type is `Function`, which describes a transformation. For 
convenience, the right-associative infix operator `**` has been overloaded to 
act as a function arrow. A function that takes multiple types can be 
[rewritten](https://en.wikipedia.org/wiki/Currying) to a sequence of 
functions.

    >>> sqrt = Real ** Real
    >>> abs = Int ** Nat

When we apply a function type to an input type, we get its output type, or, if 
the type was inappropriate, an error:

    >>> sqrt.apply(Int)
    Real
    >>> abs.apply(Real)
    ...
    Subtype mismatch. Could not satisfy:
        Real <= Int


## Schematic types and constraints

Our types are *polymorphic* in that any type is also a representative of any 
of its supertypes: the signature `Int ** Int` also applies to `UInt ** Int` 
and to `Int ** Any`. We additionally allow *parametric polymorphism* by means 
of the `TypeSchema` class, which represents all types that can be obtained by 
substituting its type variables:

    >>> compose = TypeSchema(lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ))
    >>> compose.apply(sqrt).apply(abs)
    Int ** Real

The schema is defined by an anonymous Python function whose parameters declare 
the *schematic* type variables that occur in its body. (Don't be fooled by the 
`lambda` keyword: it has little to do with lambda abstraction, and is more akin 
to universal quantification.) When the type schema is used somewhere, the 
schematic variables are automatically instantiated with concrete ones.

Often, variables in a schema cannot be just *any* type. We can use the notation 
`variable[alternatives]` to *constrain* a type to an ad-hoc typeclass --- 
meaning that the reference `variable` must be a subtype of one of the types 
specified in the `alternatives`. For instance, we might want to define a 
function that applies to both single integers and sets of integers:

    >>> f = TypeSchema(lambda α: α[Int, Set(Int)] ** α)
    >>> f.apply(Set(Nat))
    Set(Nat)

As an aside: when you need a type variable, but you don't care how it relates 
to others, you may use the *wildcard variable* `_`. The purpose goes beyond 
convenience: it communicates to the type system that it can always be a sub- 
and supertype of *anything*. (Note that it must be explicitly imported.)

    >>> from transformation_algebra import _
    >>> f = Set(_) ** Int

Typeclass constraints and wildcards can often aid in inference, figuring out 
interdependencies between types:

    >>> f = TypeSchema(lambda α, β: α[Set(β), Map(β, _)] ** β)
    >>> f.apply(Set(Int))
    Int

For large signatures, it is sometimes clearer to specify the typeclass 
constraints upfront using the `>>` notation:

    >>> f = TypeSchema(lambda α, β:
            {α[Set(β), Map(β, _)]} >> α ** β
        )

A note on type inference in the presence of subtypes. Consider that, when you 
apply a function of type `τ ** τ ** τ` to an argument with a concrete type, say 
`A`, then you can deduce that `τ >= A`. Any more specific type would be too 
restrictive. (This does *not* suggest that providing a sub-`A` *value* is 
illegal --- just that `f`'s signature should be more general.) Once all 
arguments have been supplied, `τ` can be fixed to the most specific type 
possible. This is why it's sometimes necessary to say `τ[A] ** τ ** τ` rather 
than just `A ** A ** A`: while the two are identical in what types they 
*accept*, the former can produce an *output type* that is more specific than 
`A`.

Finally, `with_parameters` is a helper function for specifying typeclasses: it 
generates type terms that contain certain parameters.

    >>> Map = TypeOperator(name="Map", params=2)
    >>> with_parameters(Map, param=Int)
    [Map(Int, _), Map(_, Int)]


## Language and expressions

Now that we know how types work, we can use them to define the operations of a 
*language*. Note that we can leave the `name` parameter blank now: it will be 
filled automatically.

    >>> lang = Language()
    >>> lang.Int = Int = TypeOperator()
    >>> lang.add = Operator(type=Int ** Int ** Int)

In fact, for convenience, you can simply incorporate *all* operators in scope 
into your language:

    >>> Int = TypeOperator()
    >>> add = Operator(type=Int ** Int ** Int)
    >>> lang = Language(scope=locals())

It is possible to define *composite* transformations: transformations that are 
derived from other, simpler ones. This should not necessarily be thought of as 
providing an *implementation*: it merely represents a decomposition into more 
primitive conceptual building blocks.

    >>> add1 = Operator(
            type=Int ** Int,
            define=lambda x: add(x, ~Int)
        )
    >>> compose = Operation(
            type=lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            define=lambda f, g, x: f(g(x))
        )

With these definitions added to `lang`, we are able to parse an expression 
using the `.parse()` function. If the result typechecks, we obtain an `Expr` 
object. This object knows its inferred type as well as that of its 
sub-expressions:

    >>> expr = lang.parse("compose add1 add1 ~Int")
    >>> expr.type
    Int
    >>> expr
    Int
     ├─Int ** Int
     │  ├─(Int ** Int) ** Int ** Int
     │  │  ├─╼ compose : (Int ** Int) ** (Int ** Int) ** Int ** Int
     │  │  └─╼ add1 : Int ** Int
     │  └─╼ add1 : Int ** Int
     └─╼ ~Int

If desired, the underlying primitive expression can be derived using 
`.primitive()`:

    >>> expr.primitive()
    Int
     ├─Int ** Int
     │  ├─╼ add : Int ** Int ** Int
     │  └─Int
     │     ├─Int ** Int
     │     │  ├─╼ add : Int ** Int ** Int
     │     │  └─╼ ~Int : Int
     │     └─╼ ~Int : Int
     └─╼ ~Int : Int


## Graphs and queries

Beyond *expressing* transformations, an additional goal of the library is to 
enable *querying* them for their constituent operations and data types.

To turn an expression into a searchable structure, we convert it to an RDF 
graph. Every data source and every operation applied to it becomes a node, 
representing the type of data that is conceptualized at that particular step in 
the transformation. Chains of nodes are thus obtained that are easily subjected 
to queries along the lines of: 'find me a transformation containing operations 
*f* and *g* that, somewhere downstream, combine into data of type *t*'.

The process is straightforward when operations only take data as input. 
However, expressions in an algebra may also take other operations, in which 
case the process is more involved; for now, consult the source code.

In practical terms, to obtain a graph representation of the previous expression, you may do:

    >>> from transformation_algebra.rdf import TransformationGraph
    >>> g = TransformationGraph()
    >>> g.add_expr(expr)
    >>> g.serialize("graph.ttl", format="ttl")

You may use the [viz.sh](tools/viz.sh) script to visualize the graph.

These graphs can be queried via constructs from the [SPARQL 1.1 specification](https://www.w3.org/TR/sparql11-query/).

