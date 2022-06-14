# Concrete types and subtypes

To specify type transformations, we first need to declare *base types*. To 
this end, we use the `TypeOperator` class.

    >>> from transformation_algebra import *
    >>> Real = TypeOperator(name="Real")

Base types may have supertypes. For instance, anything of type `Int` is also 
automatically of type `Real`, but not necessarily of type `Nat`:

    >>> Int = TypeOperator(name="Int", supertype=Real)
    >>> Nat = TypeOperator(name="Nat", supertype=Int)
    >>> Int.subtype(Real)
    True
    >>> Int.subtype(Nat)
    False

*Complex types* take other types as parameters. For example, `Set(Int)` could 
represent the type of sets of integers. This would automatically be a subtype 
of `Set(Real)`.

    >>> Set = TypeOperator(name="Set", params=1)
    >>> Set(Int).subtype(Set(Real))
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

Often, variables in a schema cannot be just *any* type. We can abuse indexing 
notation (`x[...]`) to *constrain* a type. A constraint can be a *subtype* 
constraint, written `x <= y`, meaning that `x`, once it is unified, must be a 
subtype of the given type `y`. It can also be an *elimination* constraint, 
written `x << {y, z}` meaning that `x` will be unified to a subtype one of the 
options, as soon as the alternatives have been eliminated. For instance, we 
might want to define a function that applies to both single integers and sets 
of integers:

    >>> f = TypeSchema(lambda α: α ** α [α << {Int, Set(Int)}])
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

    >>> f = TypeSchema(lambda α, β: α ** β [α << {Set(β), Map(β, _)}])
    >>> f.apply(Set(Int))
    Int

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
