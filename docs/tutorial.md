# Table of contents

1.  [Introduction](#introduction)
    1.  [Concept types](#concept-types)
    2.  [Transformation operators](#transformation-operators)
    3.  [Transformation expressions](#transformation-expressions)
    4.  [Workflow annotation](#workflow-annotation)
2.  [Internal transformations](#internal-transformations)
3.  [Type inference](#type-inference)
    1.  [Subtype polymorphism](#subtype-polymorphism)
    2.  [Parametric polymorphism](#parametric-polymorphism)
    3.  [Subtype constraints](#subtype-constraints)
    4.  [Elimination constraints](#elimination-constraints)
    5.  [Wildcard variables](#wildcard-variables)
    6.  [Top, bottom and unit types](#top-bottom-and-unit-types)
    7.  [Union and intersection types](#union-and-intersection-types)
4.  [Vocabulary](#vocabulary)
    1.  [Canonical types](#canonical-types)
    2.  [Type taxonomy](#type-taxonomy)
5.  [Composite operators](#composite-operators)
6.  [Querying](#queries)


# Introduction

`transformation_algebra` is a Python library that allows you to define a 
language for semantically describing tools or procedures as 
*transformations between concepts*. When you connect several such 
procedures into a *workflow*, the library can construct an RDF graph for 
you that describes it as a whole, automatically inferring the specific 
concept type at every step.

Throughout this manual, we will use a simplified version of the [core 
concept transformations of geographical information][cct] as a recurring 
example, as it was the motivating use case for the library. However, you 
can apply it to any other domain.

Transformation *operators* are the atomic steps with which more complex 
procedures are described. For example, if you have a tool to measure the
relative size of two objects, you could describe it in terms of 
operators like `ratio` and `size`. Alternatively, you could make a 
monolithic operator `relative_size` that describes the entire tool 
'atomically'. The amount of detail you go into is entirely up to you, 
because the operators don't need to specify anything about 
*implementation*: they instead represent *conceptual* steps at whatever 
level of granularity is suitable for your purpose.


### Concept types

The names of the operators should provide a hint to their intended 
semantics. However, they have no formal content besides their *type 
signature*. This signature indicates what sort of concepts it 
transforms. For instance, the `size` transformation might transform an 
object to a value. Concepts like objects and ordinal values are 
represented with *type operators*.

Before going into transformation operators, we should therefore 
understand how types work. Let's start by importing the library:

    >>> import transformation_algebra as ct

Base type operators can be thought of as atomic concepts. In our case, 
that could be an *object* or an *ordinal value*. They are declared using 
the `TypeOperator` class:

    >>> Obj = ct.TypeOperator()
    >>> Ord = ct.TypeOperator()

*Compound* type operators take other types as parameters. For example, 
`C(Obj)` could represent the type of collections of objects:

    >>> C = ct.TypeOperator(params=1)

A `Function` is a special compound type, and it's quite an important 
one: it describes a transformation from one type to another. For 
convenience, Python's infix operator `**` is abused to act as a function 
arrow. When we apply a function to an input, we get its output, or, if 
the input type was inappropriate, an error:

    >>> (Ord ** Obj).apply(Ord)
    Obj
    >>> (Ord ** Obj).apply(Obj)
    Type mismatch.


### Transformation operators

We will revisit types at a later point. For now, we know enough to 
create our first transformation language, containing the operators 
`ratio` and `size`. The `Operator` class will be used to declare them. 
We already mentioned that `size` could take an object and return an 
ordinal, which would look like this:

    >>> size = ct.Operator(type=Obj ** Ord)

The accompanying `ratio` operator might take two ordinal values and 
output another: the ratio between them. Knowing that a function with 
multiple arguments can be [rewritten][w:currying] into one that takes a 
single argument and returns another function to deal with the rest, we 
get:

    >>> ratio = ct.Operator(type=Ord ** (Ord ** Ord))

Since the `**` operator is right-associative, the parentheses are 
optional.

We can now bundle up our types and operators into a transformation 
language, using the `Language` class. Let us call it `stl`, for 'simple 
transformation language'.

All the types and operators we declared need to be added to it. However, 
for convenience, it is possible to simply incorporate all types and 
operators in local scope:

    >>> stl = ct.Language(scope=locals(),
            namespace="https://example.com/stl/")


### Transformation expressions

We now have an object that represents our language. We can use `stl`'s 
`.parse()` method to parse transformation *expressions*: complex 
combinations of operators. The parser accepts both function notation, 
like `f(x, y)`, and lambda-style notation, like `f x y`.

In addition to operators, the expressions may contain numbers. These 
numbers indicate concepts that are to be provided as input. 
Alternatively, a dash (`-`) can be used to refer to an anonymous input: 
a concept that is not further specified. Finally, the notation 
`expression : type` is used to explicitly indicate the type of a 
sub-expression.

As an example, the following expression represents a concept 
transformation that selects, from an unspecified set of objects, the 
object that is nearest in distance to some other unspecified object.

    >>> expr = stl.parse("ratio (size -) (size -)")

If the result typechecks, which it does, we obtain an `Expr` object. We 
can inspect its inferred type and get a representation of its 
sub-expressions:

    >>> print(expr.tree())
    Ord
     ├─Ord → Ord
     │  ├─╼ ratio : Ord → Ord → Ord
     │  └─Ord
     │     ├─╼ size : Obj → Ord
     │     └─╼ - : τ2 [τ2 <= Obj]
     └─Ord
        ├─╼ size : Obj → Ord
        └─╼ - : τ2 [τ2 <= Obj]


### Workflow annotation

It is now time to construct a simple workflow. We will use RDF for that, 
using [Turtle][w:ttl] syntax and the [Workflow][wf] vocabulary. First, 
we describe the `RelativeSize` tool, and then we describe a workflow 
that uses it. The concrete inputs in the tool application will 
correspond to the conceptual inputs in the tool's transformation 
expression.

    @prefix : <https://example.com/>.
    @prefix stl: <https://example.com/stl/>.
    @prefix wf: <http://geographicknowledge.de/vocab/Workflow.rdf#>.

    :RelativeSize stl:expression "ratio (size 1) (size 2)".

    :SimpleWorkflow a wf:Workflow;
        wf:source _:population2021, _:population2022;
        wf:edge [
            wf:applicationOf :RelativeSize;
            wf:input1 _:population2021;
            wf:input2 _:population2022;
            wf:output _:population_increase
        ].

We will now use `transformation_algebra`'s command-line interface to 
enrich the original *workflow* graph with a *transformation* graph. 
Consult `python -m transformation_algebra -h` for more information on 
how to use the command. You can also interface with Python directly; 
checking out the source code for the command-line interface should give 
you a headstart.

Save the transformation language into [`stl.py`](resource/stl.py), and 
the workflow and tool description into 
[`wf.ttl`](resource/wf-RelativeSize.ttl), and run:

    python -m transformation_algebra graph \
        -L stl.py -T wf.ttl wf.ttl -t ttl -o -

What we get back is an RDF graph in which the transformation expression 
for each tool application has been "unfolded": every input and every 
operation applied to it becomes a node, representing the concept that is 
involved at that particular step in the transformation. It will look 
something like this:

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quangis/transformation-algebra/develop/docs/resource/RelativeSize-tg.svg">
</p>

This graph tells us what happens inside the workflow *conceptually*: the 
size is an ordinal value that was derived from our inputs and considered 
before finding the ratio between them.

Of course, this is a toy problem: the workflow contains only one trivial 
tool, and the operators do not generalize well. In what follows, we will 
go into more advanced features.


# Internal transformations

Suppose we want to describe a `TallestBuilding` tool for selecting the 
tallest among a collection of buildings. Of course, we could introduce 
`highest` with a type of `C(Obj) ** Obj`, but this does not generalize 
well: you are probably going to need a slew of operators that *maximize 
something*. In such a case, we can introduce an operator that is 
parameterized by *another operator*:

    >>> maximum = ct.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)
    >>> height = ct.Operator(type=Obj ** Ord)

The expression `maximum height (- : C(Obj))` would use `height` to 
associate each building with the value to be maximized. If we were to 
unfold it into a graph, it would look like this:

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quangis/transformation-algebra/develop/docs/resource/TallestBuilding-tg.svg">
</p>

You might notice that a dotted node appeared. This is because a 
transformation that is used as a parameter is a 'black box', in that we 
haven't specified exactly how it is applied. The inner transformation 
could need access to the resources available to the outer 
transformation, and indeed, that is the case here: `height` is applied 
to the buildings in the second argument to `maximum`. The dotted node 
represents these internal operations.

In this context, it is also possible to perform *partial application*. 
Suppose that we want to describe a `NearObject` tool for finding, among 
a collection of objects `2`, the one closest to some object `1`. It is 
natural to do so in the following terms:

    >>> minimum = ct.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)
    >>> distance = ct.Operator(type=Obj ** Obj ** Ord)

This time, the `distance` transformation should already be anchored to 
object `1` as its first operand. This is no problem:

    :NearObject
        :expression "minimum (distance (1: Obj)) (2: C(Obj))".


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

What follows is work-in-progress.

# Type inference

### Subtype polymorphism

Base types may have sub- and supertypes. For instance, an ratio-scaled 
quality is also an ordinal value, but not vice versa:

    >>> Ord = ct.TypeOperator(supertype=Qlt)
    >>> Ratio = ct.TypeOperator(supertype=Ord)
    >>> Ratio.subtype(Ord)
    True
    >>> Ord.subtype(Ratio)
    False

This automatically extends to compound types:

    >>> C(Ratio).subtype(C(Ord))
    True

An operator may take another operator as an argument. An operator that 
takes

Functional types, too, may be representative of any of its 

These types are *polymorphic* in that any type is also a representative 
of any of its supertypes. That is, an operator that expects an argument 
of type `Ord ** Ord` would also accept `Qlt ** Ord` or `Ord ** Ratio`.


    >>> ((Ord ** Ord) ** Qlt).apply(Qlt ** Ord)
    Qlt
    >>> ((Ord ** Ord) ** Qlt).apply(Ord ** Ratio)
    Qlt

### Parametric polymorphism

We additionally allow *parametric polymorphism* by means of the 
`TypeSchema` class, which represents all types that can be obtained by 
substituting its type variables:

    >>> compose = ct.TypeSchema(lambda α, β, γ:
            (β ** γ) ** (α ** β) ** (α ** γ))
    >>> compose.apply(f).apply(g)
    Int ** Real

Don't be fooled by the `lambda` keyword: it has little to do with lambda 
abstraction. It is there because we use an anonymous Python function, 
whose parameters declare the *schematic* type variables that occur in 
its body. When the type schema is used somewhere, the schematic 
variables are automatically instantiated with *concrete* variables.


### Subtype constraints

Often, variables in a schema cannot be just *any* type. We can abuse 
indexing notation (`x [...]`) to *constrain* a type. A constraint can be 
a *subtype* constraint, written `x <= y`, meaning that `x`, once it is 
unified, must be a subtype of the given type `y`. 

In the presence of subtypes, type inference can be less than 
straightforward. Consider that, when you apply a function of type `τ ** 
τ ** τ` to an argument with a concrete type, say `A`, then we cannot 
immediately bind `τ` to `A`: what if the second argument to the function 
is a supertype of `A`? We can, however, deduce that `τ >= A`, since any 
more specific type would certainly be too restrictive. This does not 
suggest that providing a *value* of a more specific type is illegal --- 
just that the signature should be more general. Only once all arguments 
have been supplied can `τ` be fixed to the most specific type possible.

This is why it's sometimes necessary to say `τ ** τ ** τ [τ <= A]` 
rather than just `A ** A ** A`: while the two are identical in what 
types they *accept*, the former can produce an *output type* that is 
more specific than `A`.


### Elimination constraints

It can also be an *elimination* constraint, written `x << {y, z}`, 
meaning that `x` will be unified to a subtype one of the options as soon 
as the alternatives have been eliminated. For instance, we might want a 
function signature that applies to both single integers and sets of 
integers:

    >>> f = TypeSchema(lambda α: α ** α [α << {Int, Set(Int)}])
    >>> f.apply(Set(Nat))
    Set(Nat)

Typeclass constraints and wildcards can often aid in inference, figuring 
out interdependencies between types:

    >>> Map = ct.TypeOperator('Map', params=2)
    >>> f = TypeSchema(lambda α, β: α ** β [α << {Set(β), Map(β, _)}])
    >>> f.apply(Set(Int))
    Int


### Wildcard variables

As an aside: when you need a type variable, but you don't care how it 
relates to others, you may use the *wildcard variable* `_`. The purpose 
goes beyond convenience: it communicates to the type system that it can 
always be a sub- and supertype of *anything*. It must be explicitly 
imported:

    >>> from transformation_algebra import _
    >>> f = Set(_) ** Int


### Top, bottom and unit types

(todo)


### Union and intersection types

Just as transformation operators might not correspond to how the 
procedure is *implemented*, type operators don't necessarily represent 
*data* types. They only capture some relevant conceptual properties of 
the entities they describe. For example, when a procedure takes an 
ordinal as argument, that doesn't mean that some concrete number is 
passed to it --- just that it operates on things that are, in some 
aspect, ordinal-scaled.

*Union and intersection types have not yet been implemented.*


# Vocabulary

### Canonical types

(todo)


### Type taxonomy

(todo)


# Composite operators

It is possible to define *composite* transformations: transformations 
that are derived from other, simpler ones. This should not necessarily 
be thought of as providing an *implementation*: it merely represents a 
decomposition into more primitive conceptual building blocks.

    >>> add1 = ct.Operator(
            type=Int ** Int,
            define=lambda x: add(x, ct.Source(Int))
        )
    >>> compose = ct.Operation(
            type=lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            define=lambda f, g, x: f(g(x))
        )

When we use such composite operations, we can derive the underlying 
primitive expression using `.primitive()`:

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


# Queries

The goal of the library is not just to *express* transformations, but 
also to *query* associated workflows for their constituent operations 
and concept types.

RDF graphs are searchable structures that can be subjected to 
[SPARQL][sparql] queries along the lines of: 'find me a workflow that 
contains operations `f` and `g` that, somewhere downstream, combine into 
a concept of type `t`'.[^5]

[^5]: The process is straightforward when operations only take data as 
    input. However, expressions in an algebra may also take other 
    operations, in which case the process is more involved; for now, 
    consult the source code.

These queries can themselves be represented as (partial) transformation 
graphs. We call such partial graphs *tasks*, because they are meant to 
express what sort of conceptual properties we expect from a workflow 
that solves a particular task. For example, the following graph captures 
what we might expect of a workflow that finds the closest hospital:

    @prefix : <https://github.com/quangis/transformation-algebra#>.
    @prefix x: <https://example.com/#>.

    _:Hospitals x:type "C(Obj)".
    _:Incident x:type "Obj".
    [] a :Task;
        :input _:HospitalPoints, _:IncidentLocations;
        :output [
            x:via "minimum";
            x:type "Obj";
            :from [
                x:via "distance";
                :from _:Hospitals;
                :from _:Incident
            ]
        ].

To convert this graph into a SPARQL query, you can once again use the 
command-line interface:

    python -m transformation_algebra query -L sl.py task.ttl


[cct]: https://github.com/quangis/cct
[wf]: http://geographicknowledge.de/vocab/Workflow.rdf
[sparql]: https://www.w3.org/TR/sparql11-query/
[w:ttl]: https://en.wikipedia.org/wiki/Turtle_(syntax)
[w:currying]: https://en.wikipedia.org/wiki/Currying
