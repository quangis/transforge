# Types

An expression of a transformation language is composed of user-defined semantic 
operators. Each of those operators should be given a *type signature* to 
indicate what sort of concepts it transforms between. Before defining a 
transformation language, we should therefore understand how types work.

## Subtype polymorphism

The `TypeOperator` class is used to declare *base types*. These can be thought 
of as atomic concepts, such as the real numbers:

    >>> import transformation_algebra as ta
    >>> Real = ta.TypeOperator('Real')

Base types may have sub- and supertypes. For instance, an integer is also 
automatically a real number, but not necessarily a natural number:

    >>> Int = ta.TypeOperator('Int', supertype=Real)
    >>> Nat = ta.TypeOperator('Nat', supertype=Int)
    >>> Int.subtype(Real)
    True
    >>> Int.subtype(Nat)
    False

*Compound types* take other types as parameters. For example, `Set(Int)` could 
represent the type of sets of integers. This would automatically be a subtype 
of `Set(Real)`.

    >>> Set = ta.TypeOperator('Set', params=1)
    >>> Set(Int).subtype(Set(Real))
    True

A `Function` is a special compound type, and it's quite an important one: it 
describes a transformation. For convenience, the right-associative infix 
operator `**` has been overloaded to act as a function arrow. (Note that a 
function that takes multiple arguments can be 
[rewritten](https://en.wikipedia.org/wiki/Currying) to a sequence of 
functions.)

    >>> f = Real ** Real
    >>> g = Int ** Nat

When we apply a function type to an input type, we get its output type, or, if 
the type was inappropriate, an error:

    >>> f.apply(Int)
    Real
    >>> g.apply(Real)
    ...
    Subtype mismatch. Could not satisfy:
        Real <= Int


## Parametric polymorphism

Our types are *polymorphic* in that any type is also a representative of any of 
its supertypes. That is, an operator that expects an argument of type `Nat ** 
Nat` would also accept `Int ** Nat` or `Nat ** Real`. We additionally allow 
*parametric polymorphism* by means of the `TypeSchema` class, which represents 
all types that can be obtained by substituting its type variables:

    >>> compose = ta.TypeSchema(lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ))
    >>> compose.apply(f).apply(g)
    Int ** Real

Don't be fooled by the `lambda` keyword: it has little to do with lambda 
abstraction. It is there because we use an anonymous Python function, whose 
parameters declare the *schematic* type variables that occur in its body. When 
the type schema is used somewhere, the schematic variables are automatically 
instantiated with *concrete* variables.


## Constraints

Often, variables in a schema cannot be just *any* type. We can abuse indexing 
notation (`x [...]`) to *constrain* a type. A constraint can be a *subtype* 
constraint, written `x <= y`, meaning that `x`, once it is unified, must be a 
subtype of the given type `y`. It can also be an *elimination* constraint, 
written `x << {y, z}`, meaning that `x` will be unified to a subtype one of the 
options as soon as the alternatives have been eliminated. For instance, we 
might want a function signature that applies to both single integers and sets 
of integers:

    >>> f = TypeSchema(lambda α: α ** α [α << {Int, Set(Int)}])
    >>> f.apply(Set(Nat))
    Set(Nat)

As an aside: when you need a type variable, but you don't care how it relates 
to others, you may use the *wildcard variable* `_`. The purpose goes beyond 
convenience: it communicates to the type system that it can always be a sub- 
and supertype of *anything*. It must be explicitly imported:

    >>> from transformation_algebra import _
    >>> f = Set(_) ** Int

Typeclass constraints and wildcards can often aid in inference, figuring out 
interdependencies between types:

    >>> Map = ta.TypeOperator('Map', params=2)
    >>> f = TypeSchema(lambda α, β: α ** β [α << {Set(β), Map(β, _)}])
    >>> f.apply(Set(Int))
    Int


## Type inference

In the presence of subtypes, type inference can be less than straightforward. 
Consider that, when you apply a function of type `τ ** τ ** τ` to an argument 
with a concrete type, say `A`, then we cannot immediately bind `τ` to `A`: what 
if the second argument to the function is a supertype of `A`? We can, however, 
deduce that `τ >= A`, since any more specific type would certainly be too 
restrictive. This does not suggest that providing a *value* of a more specific 
type is illegal --- just that the signature should be more general. Only once 
all arguments have been supplied can `τ` be fixed to the most specific type 
possible.

This is why it's sometimes necessary to say `τ ** τ ** τ [τ <= A]` rather than 
just `A ** A ** A`: while the two are identical in what types they *accept*, 
the former can produce an *output type* that is more specific than `A`.
