# Language and expressions

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
