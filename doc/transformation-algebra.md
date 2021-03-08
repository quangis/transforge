# Type inference for a transformation algebra

A transformation algebra describes abstract transformations in some domain. An 
expression of such an algebra should have an interpretation, but there is not 
necessarily any concrete data structure or implementation assigned to it. The 
algebra simply describes what type of data some tool can take, and what type 
of data it produces.

To define such an algebra, we implemented a type inference system in Python. 
To make it work, some magic happens under the hood; for now, refer to the 
[source code](../quangis/transformation/type.py) to gain a deeper 
understanding. This document is merely intended to be a user's guide.


## Concrete types and subtypes

We first need to declare some bacic type signatures. For this, we use the 
`Operator` class. 

    >>> from quangis.transformation.type import Operator
    >>> Any = Operator("Any")

Basic types may have supertypes. For instance, anything of type `Int` is also 
automatically of type `Any`, but not necessarily of type `UInt`:

    >>> Int = Operator("Int", supertype=Any)
    >>> UInt = Operator("UInt", supertype=Int)
    >>> Int <= Any
    True
    >>> Int <= UInt
    False

Higher-order types take other types as parameters, allowing us to combine 
types. For example, `Set(Int)` could represent a set of integers. Note that 
this is automatically a subtype of `Set(Any)`.

    >>> Set = Operator("Set", 1)

A special higher-order type is `Function`, which describes a transformation. 
For convenience, to create functions, the right-associative infix operator 
`**` has been overloaded, resembling the function arrow. A function that takes 
multiple types can be [rewritten](https://en.wikipedia.org/wiki/Currying) to a 
sequence of functions.

    >>> add = Int ** Int ** Int
    >>> abs = Int ** UInt

When we apply an input type to a function signature, we get its output type, 
or, if the type was inappropriate, an error:

    >>> add(Int, UInt)
    Int
    >>> add(Any)
    ...
    Subtype mismatch. Could not satisfy:
        Any <= Int


## Schematic types and constraints

Our types are *polymorphic* in that any type also represents its supertypes: 
the signature `Int ** Int` also applies to `UInt ** Int` or indeed to `Int ** 
Any`. We additionally allow *parametric polymorphism*, using the `Schema` 
class.

    >>> from quangis.transformation.type import Schema
    >>> compose = Schema(lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ))
    >>> compose(abs, add(Int))
    Int ** UInt

Don't be fooled by the `lambda` keyword --- this is an implementation artefact 
and does not refer to lambda abstraction. Instead, the `Schema` is initialized 
with an anonymous Python function, its parameters declaring the variables that 
occur in its body, akin to *quantifying* those variables. When *instantiating* 
the schema, these variables are automatically populated.

Often, variables in a schema are not universally quantified, but *constrained* 
to some typeclass. We can use the `type | constraint` operator to attach 
constraints to a term, and the `type @ [types]` operator to say that a type 
must be the same as one of the given types. For instance, we might want to 
define a function that applies to both single integers and sets of integers:

    >>> sum = Schema(lambda α: α ** α | α @ [Int, Set(Int)])
    >>> sum(Set(UInt))
    Set(UInt)

Additionally, we can constrain a type to a subtype or supertype with, 
respectively, the `type << type` and `type >> type` operators.


## Algebra and expressions

Once we have created our types, we may define the `TransformationAlgebra` that 
will read expressions of the algebra.

    >>> from quangis.transformation.algebra import TransformationAlgebra
    >>> algebra = TransformationAlgebra(
            number=Int,
            abs=Int ** UInt,
            ...
    )

In fact, for convenience, you may simply define your types as globals and 
automatically create the algebra once you are done:

    >>> algebra = TransformationAlgebra.from_dict(globals())

At this point, we can parse strings using `algebra.parse`. The parser accepts 
the names of any function and input type. Adjacent expressions are applied to 
eachother and the parser will only succeed if the result typechecks. Input 
types can take an optional identifier to label that input.

    >>> algebra.parse("abs (number x)")

This will give you an `Expr` object, which has a `type` and sub-expressions.
