# Table of contents

1.  [Types](#types)
2.  [Expressions](#language-and-expressions)
3.  [Querying](#graphs-and-queries)

# Types

An expression of a transformation language is composed of user-defined semantic 
operators. Each of those operators should be given a *type signature* to 
indicate what sort of concepts it transforms between. Before defining a 
transformation language, we should therefore understand how types work.

### Subtype polymorphism

The `TypeOperator` class is used to declare *base types*. These can be thought 
of as atomic concepts, such as the real numbers:

    >>> import transformation_algebra as ta
    >>> Real = ct.TypeOperator('Real')

Base types may have sub- and supertypes. For instance, an integer is also 
automatically a real number, but not necessarily a natural number:

    >>> Int = ct.TypeOperator('Int', supertype=Real)
    >>> Nat = ct.TypeOperator('Nat', supertype=Int)
    >>> Int.subtype(Real)
    True
    >>> Int.subtype(Nat)
    False

*Compound types* take other types as parameters. For example, `Set(Int)` could 
represent the type of sets of integers. This would automatically be a subtype 
of `Set(Real)`.

    >>> Set = ct.TypeOperator('Set', params=1)
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


### Parametric polymorphism

Our types are *polymorphic* in that any type is also a representative of any of 
its supertypes. That is, an operator that expects an argument of type `Nat ** 
Nat` would also accept `Int ** Nat` or `Nat ** Real`. We additionally allow 
*parametric polymorphism* by means of the `TypeSchema` class, which represents 
all types that can be obtained by substituting its type variables:

    >>> compose = ct.TypeSchema(lambda α, β, γ:
            (β ** γ) ** (α ** β) ** (α ** γ))
    >>> compose.apply(f).apply(g)
    Int ** Real

Don't be fooled by the `lambda` keyword: it has little to do with lambda 
abstraction. It is there because we use an anonymous Python function, whose 
parameters declare the *schematic* type variables that occur in its body. When 
the type schema is used somewhere, the schematic variables are automatically 
instantiated with *concrete* variables.


### Constraints

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

    >>> Map = ct.TypeOperator('Map', params=2)
    >>> f = TypeSchema(lambda α, β: α ** β [α << {Set(β), Map(β, _)}])
    >>> f.apply(Set(Int))
    Int


### Type inference

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


# Language and expressions

Now that we have a feeling for types, we can declare operators using the 
`Operator` class, and give a type signature to each. A transformation language 
`Language` is a collection of such types and operators.

For convenience, you can simply incorporate all types and operators in scope 
into your language. In this case, we also no longer need to provide a name for 
the types: it will be filled automatically. A very simple language, containing 
two types and one operator, could look as follows:

    >>> Int = ct.TypeOperator()
    >>> Nat = ct.TypeOperator(supertype=Int)
    >>> add = ct.Operator(type=lambda α: α ** α ** α [α <= Int])
    >>> lang = ct.Language(scope=locals())

We can immediately parse expressions of this language using the `.parse()` 
method. For example, the following expression adds some unspecified input of 
type `Nat` (written `- : Nat`) to another input of type `Int`.

    >>> expr = lang.parse("add (- : Nat) (- : Int)")

If the result typechecks, which it does, we obtain an `Expr` object. We can 
inspect its inferred type:

    >>> expr.type
    Int

We can also get a representation of its sub-expressions:

    >>> print(expr.tree())
    Int
     ├─Int ** Int
     │  ├─╼ add : Int ** Int ** Int
     │  └─╼ - : Nat
     └─╼ - : Int


### Composite operators

It is possible to define *composite* transformations: transformations that are 
derived from other, simpler ones. This should not necessarily be thought of as 
providing an *implementation*: it merely represents a decomposition into more 
primitive conceptual building blocks.

    >>> add1 = ct.Operator(
            type=Int ** Int,
            define=lambda x: add(x, ct.Source(Int))
        )
    >>> compose = ct.Operation(
            type=lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            define=lambda f, g, x: f(g(x))
        )

When we use such composite operations, we can derive the underlying primitive 
expression using `.primitive()`:

    >>> expr = lang.parse("compose add1 add1 (-: Nat)")
    >>> print(expr.primitive().tree())
    Int
     ├─Int ** Int
     │  ├─╼ add : Int ** Int ** Int
     │  └─Int
     │     ├─Int ** Int
     │     │  ├─╼ add : Int ** Int ** Int
     │     │  └─╼ - : Int
     │     └─╼ - : Int
     └─╼ - : Nat


# Graphs and queries

Beyond *expressing* transformations, an additional goal of the library is to 
enable *querying* them for their constituent operations and data types.

To turn an expression into a searchable structure, we convert it to an RDF 
graph. Every data source and every operation applied to it becomes a node, 
representing the type of data that is conceptualized at that particular step in 
the transformation. Chains of nodes are thus obtained that are easily subjected 
to queries along the lines of: 'find me a transformation containing operations 
`f` and `g` that, somewhere downstream, combine into data of type `t`'.

The process is straightforward when operations only take data as input. 
However, expressions in an algebra may also take other operations, in which 
case the process is more involved; for now, consult the source code.

In practical terms, to obtain a graph representation of the previous 
expression, you may do:

    >>> g = ct.TransformationGraph()
    >>> g.add_expr(expr)
    >>> g.serialize("graph.ttl", format="ttl")

These graphs can be queried via constructs from the [SPARQL 1.1 
specification](https://www.w3.org/TR/sparql11-query/).

