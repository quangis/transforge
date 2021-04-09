# transformation-algebra

[![](https://img.shields.io/pypi/v/transformation-algebra)](https://pypi.org/project/transformation-algebra/)

A transformation algebra describes *tools* (in some domain) as *abstract data 
transformations*. An expression of such an algebra has an interpretation, but 
there is not necessarily any concrete data structure or implementation 
assigned to it. It merely describes its input and output in terms of some 
*conceptual* properties that are deemed relevant.

To define such an algebra, we implemented **type inference** in Python. The 
system accommodates both [subtype](https://en.wikipedia.org/wiki/Subtyping) 
and [parametric 
polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism), which, 
divorced from implementation, enable powerful type inference. To make it work, 
some magic happens under the hood --- for now, refer to the [source 
code](https://github.com/quangis/transformation_algebra/blob/master/transformation_algebra/type.py) 
to gain a deeper understanding. This document is merely intended to be a 
user's guide.

The library was developed for the core concept transformation algebra 
([CCT](https://github.com/quangis/cct)) for geographical information, which 
can act as an example.


## Concrete types and subtypes

To specify type transformations, we first need to declare *base types*. To 
this end, we use the `Type.declare` function. 

    >>> from transformation_algebra import *
    >>> Any = Type.declare("Any")

Base types may have supertypes. For instance, anything of type `Int` is also 
automatically of type `Any`, but not necessarily of type `UInt`:

    >>> Int = Type.declare("Int", supertype=Any)
    >>> UInt = Type.declare("UInt", supertype=Int)
    >>> Int <= Any
    True
    >>> Int <= UInt
    False

*Complex types* take other types as parameters. For example, `Set(Int)` could 
represent a set of integers. This would automatically be a subtype of 
`Set(Any)`.

    >>> Set = Type.declare("Set", params=1)
    >>> Set(Int) <= Set(Any)
    True

A special complex type is `Function`, which describes a transformation. For 
convenience, to create functions, the right-associative infix operator `**` 
has been overloaded, resembling the function arrow. A function that takes 
multiple types can be [rewritten](https://en.wikipedia.org/wiki/Currying) to a 
sequence of functions.

    >>> add = Int ** Int ** Int
    >>> abs = Int ** UInt

When we apply an input type to a function signature, we get its output type, 
or, if the type was inappropriate, an error:

    >>> add.apply(Int).apply(UInt)
    Int
    >>> add.apply(Any)
    ...
    Subtype mismatch. Could not satisfy:
        Any <= Int


## Schematic types and constraints

Our types are *polymorphic* in that any type also represents its supertypes: 
the signature `Int ** Int` also applies to `UInt ** Int` or indeed to `Int ** 
Any`. We additionally allow *parametric polymorphism*, using the `TypeSchema` 
class.

    >>> compose = TypeSchema(lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ))
    >>> compose.apply(abs).apply(add.apply(Int))
    Int ** UInt

Don't be fooled by the `lambda` keyword --- this is an implementation artefact 
and does not refer to lambda abstraction. Instead, the `TypeSchema` is 
initialized with an anonymous Python function whose parameters declare the 
variables that occur in its body. This is akin to *quantifying* those 
variables. When instantiating the schema, these generic variables are 
automatically populated with *instances* of type variables.

When you need a variable, but you don't care how it relates to others, you may 
use the *wildcard variable* `_`. The purpose goes beyond convenience: it 
communicates to the type system that it can always be a sub- and supertype of 
*anything*. (Note that it must be explicitly imported.)

    >>> from transformation_algebra import _
    >>> size = Set(_) ** Int

Often, variables in a schema are not universally quantified, but *constrained* 
to some typeclass. We can use the `t | c` operator to attach a typeclass 
constraint `c` to a term `t`. `c` can then specify the typeclass in an ad-hoc 
manner, using the `t @ [ts]` operator (meaning that a term `t` must be a 
subtype of one of the types specified in `[ts]`). For instance, we might want 
to define a function that applies to both single integers and sets of 
integers:

    >>> sum = TypeSchema(lambda α: α ** α | α @ [Int, Set(Int)])
    >>> sum.apply(Set(UInt))
    Set(UInt)

Typeclass constraints can often aid in inference, figuring out 
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

    >>> example = TransformationAlgebra(
            one=Data(Int),
            add=Operation(Int ** Int ** Int)
            ...
    )

In fact, for convenience, you may provide your definitions globally and 
automatically incorporate them into the algebra:

    >>> one = Data(Int)
    >>> add = Operation(Int ** Int ** Int)
    >>> example = TransformationAlgebra(**globals())

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

With these definitions added to `example`, we are able to parse an expression 
using the `TransformationAlgebra.parse()` function. If the result typechecks, 
we obtain an `Expr` object. This object knows its inferred type as well as 
that of its sub-expressions:

    >>> e = example.parse("compose add1 add1 one")
    >>> e.type
    Int
    >>> print(e.tree())
    Int
     ├─Int ** Int
     │  ├─(Int ** Int) ** Int ** Int
     │  │  ├─╼ compose : (β ** γ) ** (α ** β) ** α ** γ
     │  │  └─╼ add1 : Int ** Int
     │  └─╼ add1 : Int ** Int
     └─╼ one : Int

If desired, the underlying primitive expression can be derived using 
`Expr.primitive()`:

    >>> print(e.primitive().tree())
    Int
     ├─Int ** Int
     │  ├─╼ add : Int ** Int ** Int
     │  └─Int
     │     ├─Int ** Int
     │     │  ├─╼ add : Int ** Int ** Int
     │     │  └─╼ one : Int
     │     └─╼ one : Int
     └─╼ one : Int
