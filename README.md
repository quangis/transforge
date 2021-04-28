# transformation-algebra

[![](https://img.shields.io/pypi/v/transformation-algebra)](https://pypi.org/project/transformation-algebra/)

A transformation algebra is a notational system for describing *tools* as 
abstract semantic *transformations*. The expressions of such an algebra are 
not necessarily associated with any concrete data structure or implementation: 
they merely describe input and output in terms of some *conceptual properties* 
that are deemed relevant.

To enable reasoning about its expressions, we implemented [type 
inference](https://en.wikipedia.org/wiki/Type_inference) in Python. While the 
algorithm is probably also useful in the more traditional context of catching 
implementation errors, it was written to be separate from such concerns. It 
accommodates both [subtype](https://en.wikipedia.org/wiki/Subtyping) and 
[parametric 
polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism).

For now, refer to the [source 
code](https://github.com/quangis/transformation_algebra/blob/master/transformation_algebra/type.py) 
to gain a deeper understanding. This document is merely intended to be a 
user's guide. The library was developed for the core concept transformation 
algebra for geographical information ([CCT](https://github.com/quangis/cct)) — 
code in that repository may act as an example.


## Concrete types and subtypes

To specify type transformations, we first need to declare *base types*. To 
this end, we use the `Type.declare` function. 

    >>> from transformation_algebra import *
    >>> Real = Type.declare("Real")

Base types may have supertypes. For instance, anything of type `Int` is also 
automatically of type `Real`, but not necessarily of type `Nat`:

    >>> Int = Type.declare("Int", supertype=Real)
    >>> Nat = Type.declare("Nat", supertype=Int)
    >>> Int <= Real
    True
    >>> Int <= Nat
    False

*Complex types* take other types as parameters. For example, `Set(Int)` could 
represent the type of sets of integers. This would automatically be a subtype 
of `Set(Real)`.

    >>> Set = Type.declare("Set", params=1)
    >>> Set(Int) <= Set(Real)
    True

A special complex type is `Function`, which describes a transformation. For 
convenience, the right-associative infix operator `**` has been overloaded to 
act as a function arrow. A function that takes multiple types can be 
[rewritten](https://en.wikipedia.org/wiki/Currying) to a sequence of 
functions.

    >>> sqrt = Real ** Real
    >>> abs = Int ** Nat

When we apply an input type to a function signature, we get its output type, 
or, if the type was inappropriate, an error:

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
the type variables that occur in its body. This is akin to quantifying those 
variables (don't be fooled by the `lambda` keyword — it has little to do with 
lambda abstraction!). When instantiated, the *schematic* variables are 
automatically populated with concrete *instances* of type variables.

Often, variables in a schema are *constrained* to some typeclass. We can use 
the notation `context | type @ alternatives` notation to constrain a type to 
some ad-hoc typeclass --- meaning that the `type` must be a subtype of one of 
the types specified in the `alternatives`. For instance, we might want to 
define a function that applies to both single integers and sets of integers:

    >>> f = TypeSchema(lambda α: α ** α | α @ [Int, Set(Int)])
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

    >>> f = TypeSchema(lambda α, β: α ** β | α @ [Set(β), Map(_, β)])
    >>> f.apply(Set(Int))
    Int

Finally, `operators` is a helper function for specifying typeclasses: it 
generates type terms that contain certain parameters.

    >>> Map = Type.declare("Map", params=2)
    >>> operators(Map, param=Int)
    [Map(Int, _), Map(_, Int)]


## Algebra and expressions

Now that we know how types work, we can use them to define the data inputs and 
operations of an algebra.

    >>> alg = TransformationAlgebra(
            one=Data(Int),
            add=Operation(Int ** Int ** Int)
            ...
    )

In fact, for convenience, you may provide your definitions globally and 
automatically incorporate them into the algebra:

    >>> one = Data(Int)
    >>> add = Operation(Int ** Int ** Int)
    >>> alg = TransformationAlgebra(**globals())

It is possible to define *composite* transformations: transformations that are 
derived from other, simpler ones. This definition should not necessarily be 
thought of as an *implementation*: it merely represents a decomposition into 
more primitive conceptual building blocks.

    >>> add1 = Operation(
            Int ** Int,
            derived=lambda x: add(x, one)
        )
    >>> compose = Operation(
            lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            derived=lambda f, g, x: f(g(x))
        )

With these definitions added to `alg`, we are able to parse an expression 
using the `.parse()` function. If the result typechecks, we obtain an `Expr` 
object. This object knows its inferred type as well as that of its 
sub-expressions:

    >>> expr = alg.parse("compose add1 add1 one")
    >>> expr.type
    Int
    >>> expr
    Int
     ├─Int ** Int
     │  ├─(Int ** Int) ** Int ** Int
     │  │  ├─╼ compose : (Int ** Int) ** (Int ** Int) ** Int ** Int
     │  │  └─╼ add1 : Int ** Int
     │  └─╼ add1 : Int ** Int
     └─╼ one : Int

If desired, the underlying primitive expression can be derived using 
`.primitive()`:

    >>> expr.primitive()
    Int
     ├─Int ** Int
     │  ├─╼ add : Int ** Int ** Int
     │  └─Int
     │     ├─Int ** Int
     │     │  ├─╼ add : Int ** Int ** Int
     │     │  └─╼ one : Int
     │     └─╼ one : Int
     └─╼ one : Int
