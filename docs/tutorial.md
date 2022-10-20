# Table of contents

1.  [Introduction](#introduction)
    1.  [Concept types](#concept-types)
    2.  [Transformation operators](#transformation-operators)
    3.  [Workflow annotation](#workflow-annotation)
2.  [Internal operators](#internal-operators)
3.  [Type inference](#type-inference)
    1.  [Subtype polymorphism](#subtype-polymorphism)
    2.  [Parametric polymorphism](#parametric-polymorphism)
    3.  [Subtype constraints](#subtype-constraints)
    4.  [Elimination constraints](#elimination-constraints)
    5.  [Wildcard variables](#wildcard-variables)
3.  [Composite operators](#composite-operators)
4.  [Querying](#queries)


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
procedures are described. For example, if you have a `NearObject` tool 
for selecting the nearest object, you could describe it in terms of 
operators like `minimum` and `distance`. Alternatively, you could 
declare a monolithic operator `select_nearest_object` that describes the 
entire tool 'atomically'. The amount of detail you go into is entirely 
up to you, because the operators don't necessarily specify anything 
about *implementation*: they instead represent *conceptual* steps at 
whatever level of granularity is suitable for your purpose.


### Concept types

The names of the operators should provide a hint to their intended 
semantics. However, they have no formal content besides their *type 
signature*. This signature indicates what sort of concepts it 
transforms. For instance, the `distance` transformation might transform 
two *objects* to a *value*. Such concepts are represented with *type 
operators*. So, before going into transformation operators, we should 
understand how these work.

Let's start by importing the library:

    >>> import transformation_algebra as ct

[^1]: Just as with transformation operators, type operators don't 
    necessarily correspond to concrete data types --- they only capture 
    some important conceptual properties. That is, when a procedure 
    involves a `Ratio` concept, that also covers `Object`s that happen 
    to have a `Ratio` property.

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
convenience, the right-associative infix operator `**` has been 
overloaded to act as a function arrow. When we apply a function to an 
input, we get its output, or, if the input type was inappropriate, an 
error:

    >>> (Ord ** Obj).apply(Ord)
    Obj
    >>> (Ord ** Obj).apply(Obj)
    Type mismatch.


### Transformation operators

We will revisit types at a later point. For now, we know enough to be 
able to create our first simple transformation language, containing the 
operators `minimum` and `distance`. We already mentioned that a 
`distance` operator would perhaps take two objects and output a 
value.[^2] The accompanying `minimum` operator might take a set of 
objects, along with a transformation that associates an object with an 
ordinal value, and outputs the smallest object. Using the `Operator` 
class to declare transformation operators, we would end up with:

    >>> distance = ct.Operator(type=Obj ** Obj ** Ord)
    >>> minimum = ct.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)

[^2]: A function that takes multiple arguments can be 
    [rewritten][w:currying] to a sequence of functions.

Now, we bundle up our types and operators into a transformation language 
with the `Language` class. For convenience, you can simply incorporate 
all types and operators in local scope:

    >>> sl = ct.Language(scope=locals())

Now we have an object that represents the language, which we have called 
`sl`. We can now use the language's `.parse()` method to parse 
transformation *expressions*: complex combinations of operators.

In addition to operators, such expressions may contain numbers. These 
numbers indicate concepts that are to be provided as input. 
Alternatively, a dash (`-`) may be used to indicate an concept that is 
not further specified. Finally, the notation `expression : type` is used 
to explicitly indicate the type of a sub-expression.

For example, the following expression represents a concept 
transformation that selects, from an unspecified set of objects, the 
object that is nearest in distance to some other unspecified object.

    >>> expr = sl.parse("minimum (distance (- : Obj)) (- : C(Obj))")

If the result typechecks, which it does, we obtain an `Expr` object. We 
can inspect its inferred type and get a representation of its 
sub-expressions:

    >>> print(expr.tree())
    Obj
     ├─C(Obj) → Obj
     │  ├─╼ minimum : (Obj → Ord) → C(Obj) → Obj
     │  └─Obj → Ord
     │     ├─╼ distance : Obj → Obj → Ord
     │     └─╼ - : Obj
     └─╼ - : C(Obj)


### Workflow annotation

It is now time to construct a simple workflow. We will use RDF for that, 
using [Turtle][w:ttl] syntax and the [Workflow][wf] vocabulary. First, 
we describe the `NearObject` tool.

    @prefix : <http://example.com/#>.
    :NearObject
        :expression "minimum (distance (1 : Obj)) (2 : C(Obj))".

Now, we describe a workflow that uses this tool to find the hospital 
that is nearest to some incident. Note that the concrete inputs 
correspond to the conceptual inputs in the tool's transformation 
expression.

    @prefix wf: <http://geographicknowledge.de/vocab/Workflow.rdf#>.

    :SimpleWorkflow a wf:Workflow;
        wf:source _:hospitals, _:incident;
        wf:edge [
            wf:applicationOf :NearObject;
            wf:input1 _:incident;
            wf:input2 _:hospitals;
            wf:output _:nearest_hospital
        ].

Save the transformation language into [`sl.py`](resource/sl.py), and the 
workflow and tool description into [`wf.ttl`](resource/wf.ttl). We can 
now use `transformation_algebra`'s command-line interface[^3] to enrich
the original *workflow* graph with a *transformation* graph.

    python -m transformation_algebra graph \
        -L sl.py -T wf.ttl wf.ttl -o output.ttl -t ttl

What we get back is a graph in which the transformation expression for 
each tool application has been "unfolded": every input and every 
operation applied to it becomes a node, representing the concept that is 
involved at that particular step in the transformation. In this case, it 
will look something like this:

<p align="center" width="100%">
<img
    src="https://raw.githubusercontent.com/quangis/transformation-algebra/develop/docs/resource/tg.svg"
    alt="A visualization of the transformation graph.">
</p>

The distance is an ordinal value that was considered before finding the 
minimum. In other words, this graph tells us what happens inside the 
workflow *conceptually*.[^4]

Of course, the example is trivial: there is only one tool, and the 
operators do not generalize well. In what follows, we will go into more 
advanced features.

[^3]: Consult `python -m transformation_algebra -h` for more information 
    on how the command-line interface works. You can also interface with 
    Python directly; checking out the source code for the command-line 
    interface should give you a headstart.

[^4]: You might notice that there is a dashed node before the distance. 
    This is due to the fact that `distance` was an argument to another 
    transformation, so it may internally use information that was passed 
    to `minimum`, which indeed it does: `distance` needs to know about 
    the incident location as well as the hospital locations.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

What follows is work-in-progress.

# Internal operators

(to-do)


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
