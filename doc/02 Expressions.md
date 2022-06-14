# Language and expressions

Now that we have a feeling for types, we can declare operators using the 
`Operator` class, and give a type signature to each. A transformation language 
`Language` is a collection of such types and operators.

For convenience, you can simply incorporate all types and operators in scope 
into your language. In this case, we also no longer need to provide a name for 
the types: it will be filled automatically. A very simple language, containing 
two types and one operator, could look as follows:

    >>> Int = ta.TypeOperator()
    >>> Nat = ta.TypeOperator(supertype=Int)
    >>> add = ta.Operator(type=lambda α: α ** α ** α [α <= Int])
    >>> lang = ta.Language(scope=locals())

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


## Composite operators

It is possible to define *composite* transformations: transformations that are 
derived from other, simpler ones. This should not necessarily be thought of as 
providing an *implementation*: it merely represents a decomposition into more 
primitive conceptual building blocks.

    >>> add1 = ta.Operator(
            type=Int ** Int,
            define=lambda x: add(x, ta.Source(Int))
        )
    >>> compose = Operation(
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
