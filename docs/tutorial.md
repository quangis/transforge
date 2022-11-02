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
    6.  [Type aliases](#type-aliases)
    7.  [Top, bottom and unit types](#top-bottom-and-unit-types)
    8.  [Product, intersection and union types](#product-intersection-and-union-types)
4.  [Composite operators](#composite-operators)
5.  [Querying](#queries)
6.  [Canonical types](#canonical-types)


# Introduction

`transforge` is a Python library that allows you to define a language 
for semantically describing tools or procedures as *transformations 
between concepts*. When you connect several such procedures into a 
*workflow*, the library can construct an RDF graph for you that 
describes it as a whole, automatically inferring the specific concept 
type at every step.

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

    >>> import transforge as tf

Base type operators can be thought of as atomic concepts. In our case, 
that could be an *object* or an *ordinal value*. They are declared using 
the `TypeOperator` class:

    >>> Obj = tf.TypeOperator()
    >>> Ord = tf.TypeOperator()

*Compound* type operators take other types as parameters. For example, 
`C(Obj)` could represent the type of collections of objects:

    >>> C = tf.TypeOperator(params=1)

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

    >>> size = tf.Operator(type=Obj ** Ord)

The accompanying `ratio` operator might take two ordinal values and 
output another: the ratio between them. Knowing that a function with 
multiple arguments can be [rewritten][w:currying] into one that takes a 
single argument and returns another function to deal with the rest, we 
get:

    >>> ratio = tf.Operator(type=Ord ** (Ord ** Ord))

Since the `**` operator is right-associative, the parentheses are 
optional.

We can now bundle up our types and operators into a transformation 
language, using the `Language` class. Let us call it `stl`, for 'simple 
transformation language'.

All the types and operators we declared need to be added to it. However, 
for convenience, it is possible to simply incorporate all types and 
operators in local scope:

    >>> stl = tf.Language(scope=locals(),
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

We will now use `transforge`'s command-line interface to enrich the 
original *workflow* graph with a *transformation* graph. Consult 
`transforge -h` for more information on how to use the command. You can 
also interface with Python directly; checking out the source code for 
the command-line interface should give you a headstart.

Save the transformation language into [`stl.py`](resource/stl.py), and 
the workflow and tool description into 
[`wf.ttl`](resource/wf-RelativeSize.ttl), and run:

    transforge graph \
        -L stl.py -T wf.ttl wf.ttl -t ttl -o -

What we get back is an RDF graph in which the transformation expression 
for each tool application has been "unfolded": every input and every 
operation applied to it becomes a node, representing the concept that is 
involved at that particular step in the transformation. It will look 
something like this:

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quangis/transforge/develop/docs/resource/RelativeSize-tg.svg">
</p>

This graph tells us what happens inside the workflow *conceptually*: the 
size is an ordinal value that was derived from our inputs and considered 
before finding the ratio between them.

Of course, this is a toy problem: the workflow contains only one trivial 
tool, with operators that can only be combined in one way. In what 
follows, we will go into more advanced features.


# Internal transformations

Suppose we want to describe a `TallestBuilding` tool for selecting the 
tallest among a collection of buildings. Of course, we could introduce 
`highest` with a type of `C(Obj) ** Obj`, but this does not generalize 
well: you are probably going to need a slew of operators that *maximize 
something*. In such a case, we can introduce an operator that is 
parameterized by *another operator*:

    >>> maximum = tf.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)
    >>> height = tf.Operator(type=Obj ** Ord)

The expression `maximum height (- : C(Obj))` would use `height` to 
associate each building with the value to be maximized. If we were to 
unfold it into a graph, it would look like this:

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quangis/transforge/develop/docs/resource/TallestBuilding-tg.svg">
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

    >>> minimum = tf.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)
    >>> distance = tf.Operator(type=Obj ** Obj ** Ord)

This time, the `distance` transformation should already be anchored to 
object `1` as its first operand. This is no problem:

    :NearObject
        :expression "minimum (distance (1: Obj)) (2: C(Obj))".


# Type inference

In the previous section, we have seen only transformations where the 
input- and output types were fixed ahead of time.

The information that gives us is quite limited, besides ensuring that 
the output of one transformation fits the input of the next. Because the 
input and output types are always the same, an expression that uses 
transformations of this sort can only describe a specific interpretation 
of a tool. If we wanted to use a transformation in another type context, 
we would need to make a another variation of it, with a type signature 
that is subtly different. Needless to say, this does not scale.

Fortunately, the library supports *type inference*, allowing it to 
handle *generic* transformations. Such transformations may accept more 
than one input type, and the library can figure out the appropriate 
output type for you.


### Subtype polymorphism

The first way in which *polymorphism* is supported is *subtyping*: any 
type is also a representative any of its supertypes. For instance, a 
ratio-scaled quality is also an ordinal quality:

    >>> Qlt = tf.TypeOperator()
    >>> Ord = tf.TypeOperator(supertype=Qlt)
    >>> Ratio = tf.TypeOperator(supertype=Ord)
    >>> Ratio.subtype(Ord)
    True

But not vice versa:

    >>> Ord.subtype(Ratio)
    False

Subtypes can only be attached to base types, and any base type may have 
at most one supertype. This is then automatically extended to compound 
types:

    >>> C(Ratio).subtype(C(Ord))
    True

Consequently, a function that accepts values of type `Ord` will also 
accept values of the more specific type `Ratio`:

    >>> (Ord ** Ord).apply(Ratio)
    Ord
    >>> (Ord ** Ord).apply(Qlt)
    Type mismatch.

Function types, in turn, have sub- and supertypes themselves. Consider 
that a function that produces `Ord` values would, by extension, also 
produce values of the more general type `Qlt`. So, for an operator that 
takes as argument another operator of type `Ord ** Ord`, we get the 
following behaviour:

    >>> (Ord ** Ord).subtype(Qlt ** Ord)
    True
    >>> (Ord ** Ord).subtype(Ord ** Ratio)
    True
    >>> (Ord ** Ord).subtype(Ratio ** Ord)
    False
    >>> (Ord ** Ord).subtype(Ord ** Qlt)
    False


### Parametric polymorphism

What if we want to make `minimum` work on collections of things other 
than `Obj`s? With subtyping, we could introduce a universal supertype 
`Val` and change `minimum`s type signature accordingly:

    >>> Val = tf.TypeOperator()
    >>> Obj = tf.TypeOperator(supertype=Val)
    >>> Qlt = tf.TypeOperator(supertype=Val)
    >>> minimum = tf.Operator(type=(Val ** Ord) ** C(Val) ** Val)

However, we have now lost information on the relationship between the 
types of arguments. You could pass a transformation that operates on 
`Qlt`s along with a collection of `Obj`s, and `minimum` would be none 
the wiser. In the end, it will always return a generic `Val`, no matter 
what.

Therefore, we additionally allow *parametric polymorphism* by means of a 
type schema. A type schema represents all types that can be obtained by 
substituting its variables. In our case:

    >>> minimum = tf.Operator(type=lambda α: (α ** Ord) ** C(α) ** α)

Don't be fooled by the `lambda` keyword: it has little to do with lambda 
abstraction. It is there because we use an anonymous Python function, 
whose parameters declare the *schematic* type variables that occur in 
its body. When the type schema is used somewhere, the schematic 
variables are automatically instantiated with *concrete* variables. The 
concrete variables are then bound to type operators as soon as possible.

For example, once we pass arguments to the generic `minimum`, it's 
possible to figure out what the output type is supposed to be:

    >>> minimum.apply(distance.apply(tf.Source(Obj))).apply(tf.Source(C(Obj))).type Obj

In order to appreciate how this all falls into place, let's take a 
moment to use this in a workflow. We make a workflow that uses our 
`NearObject` tool to find the hospital closest to some incident, and 
inspect the resulting transformation graph:

    @prefix : <https://example.com/>
    @prefix stl: <https://example.com/stl/>.
    @prefix wf: <http://geographicknowledge.de/vocab/Workflow.rdf#>.
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.

    :NearObject
        stl:expression "minimum (distance 1) 2".

    :Workflow a wf:Workflow;
        wf:source _:Hospitals, _:Incident;
        wf:edge [
            wf:applicationOf :NearObject;
            wf:input1 _:Incident;
            wf:input2 _:Hospitals;
            wf:output _:NearestHospital
        ].

While we're at it, let's also change the type of the `distance` operator 
to produce a `Ratio`.

    >>> distance = tf.Operator(type=Obj ** Obj ** Ratio)

`minimum` will still accept this interpretation of `distance`, because 
`Ratio <= Ord`. The resulting transformation graph looks as follows --- 
the library has figured out the correct types.

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quangis/transforge/develop/docs/resource/NearObject-tg.svg">
</p>


### Subtype constraints

In the joint presence of type parameters and subtypes, inference can be 
less than straightforward. Consider that, when you apply a function of 
type `τ ** τ ** τ` to an argument with a concrete type, say `Ord`, then 
the type system cannot immediately fix `τ` to `Ord`: what if the second 
argument to the function is a `Qlt`? It can, however, deduce that `τ >= 
Ord`, since any more specific type would certainly be too restrictive. 
This does not suggest that providing a *value* of a more specific type 
is illegal --- just that the signature has to be more general. Only once 
all arguments have been supplied can `τ` be fixed to the most specific 
type possible.

The above is not only relevant to the type system: sometimes *you*, as a 
language author, want to use a schematic type in which the variables 
cannot be just *any* type. Rather than just `Ord ** Ord ** Ord`, you 
might want `τ ** τ ** τ` where `τ <= Ord`. While the two are identical 
in what types they *accept*, the former can produce an *output type* 
that is more specific than `Ord`.

To facilitate this use case, we can use *constraints*, for which 
indexing notation (`x [...]`) is abused. A *subtype* constraint is 
written `x <= y`, meaning that `x`, once bound, must be a subtype of the 
given type `y`. For example:

    >>> smallest = tf.Operator(lambda α: α ** α ** α [α <= Ord])
    >>> smallest.apply(tf.Source(Ratio)).apply(tf.Source(Ratio)).type
    Ratio
    >>> smallest.apply(tf.Source(Ratio)).apply(tf.Source(Ord)).type
    Ord
    >>> smallest.apply(tf.Source(Ratio)).apply(tf.Source(Qlt)).type
    Type mismatch.


### Elimination constraints

A constraint can also be an *elimination* constraint, written `x << {y, 
z}`, meaning that `x` will be bound to (a subtype of) one of the options 
as soon as the alternatives have been eliminated. For instance, we might 
want a function signature that applies to both single qualities and 
collections:

    >>> f = tf.Operator(type=lambda α: α ** α [α << {Qlt, C(Qlt)}])
    >>> f.apply(tf.Source(C(Ord))).type
    C(Ord)


### Wildcard variables

When you need a type variable, but you don't care how it relates to 
others, you may use the *wildcard variable* `_`. The purpose goes beyond 
convenience: it communicates to the type system that it can always be a 
sub- and supertype of *anything*. It must be explicitly imported:

    >>> from transforge import _
    >>> (C(_) ** Obj).apply(C(Ord))
    Obj

Elimination constraints and wildcards can often aid in inference, 
figuring out interdependencies between types:

    >>> R = tf.TypeOperator(params=2)
    >>> keys = tf.Operator(
            type=lambda α, β: α ** C(β) [α << {C(β), R(β, _)}])
    >>> keys.apply(tf.Source(R(Ord, Obj)))
    C(Ord)


### Type aliases

Complex types may be given alternate names. Such `TypeAlias`es may 
themselves be parameterized.

    >>> Collection = tf.TypeAlias(C)
    >>> Collection(Ord)
    C(Ord)
    >>> Ternary = tf.TypeAlias(lambda x: x ** x ** x ** x)
    >>> Ternary(Ord)
    Ord ** Ord ** Ord ** Ord


### Top, bottom and unit types

Three base types are built-in because of the special properties they 
have. First, we meet the universal type `Top`. This type contains all 
possible values, and therefore it is a supertype of everything.

    >>> from transforge import Top
    >>> (C(Top) ** Obj).apply(C(Ord))
    Obj
    >>> (C(Top) ** Obj).apply(C(C(Ord)))
    Obj
    >>> (C(Top) ** Obj).apply(Ord)
    Type mismatch.

Its evil twin is the `Bottom` type: the type that contains *no* values 
and that is a *subtype* of everything.

    >>> from transforge import Bottom
    >>> (C(Ord) ** Obj).apply(C(Bottom))
    Obj
    >>> (C(Bottom) ** Obj).apply(C(Ord))
    Type mismatch.

`Top` and `Bottom` can be used as technical tools when the type 
signature should be permissive, or when you need a concrete type without 
wildcard variables. Note also the difference between `Top` and `_`:

    >>> (Obj ** Obj).apply(_)
    Obj
    >>> (Obj ** Obj).apply(Top)
    Type mismatch.

In between `Top` and `Bottom` sits the `Unit` type. This type has one 
unique value, like a 0-tuple. To appreciate its value as a technical 
tool, imagine that you have modelled mappings between types using the 
`R` operator (as is the case for the [CCT][cct] algebra):

    >>> R = tf.TypeOperator(params=2)

Now, when every value is mapped to the same element, it essentially 
turns into a set. Instead of having a dedicated `C` operator, we can 
make it an alias.

    >>> C = tf.TypeAlias(lambda x: R(x, Unit))

The benefit is that transformations on relations will automatically also 
work on collections. For example, we can model regions as *collections 
of locations* or as *relations between locations and booleans* (ie. 
boolean fields), and they will both match the type `R(Loc, _)`.


### Product, intersection and union types

Three additional compound types are built-in. To *bundle* multiple 
types, you can use the `Product` type. It is written `A * B`.

    >>> first = tf.Operator(type=lambda α: (α * _) ** α)
    >>> first.apply(tf.Source(Ratio * Obj)).type
    Ratio

When you obtain a `Product` from a transformation, that means you got 
multiple 'things', each with their own type. However, even *one* thing 
might have multiple types.

Just as transformation operators might not correspond to how a the 
procedure is *implemented*, type operators don't necessarily represent 
*data*. They only capture some relevant conceptual properties of the 
entities they describe. For example, when a transformation takes an 
ratio as argument, that doesn't mean that a concrete number is passed to 
it --- just that it operates on things that are, in some aspect, 
ratio-scaled.

Therefore, the intersection of the set of rational values and objects is 
not empty --- it just contains the entities that *attributes* of both. 
If your transformation operates on the intersection of `A` and `B`, we 
use an `Intersection` type, written `A & B`. If it operates on things 
that have attributes of *either* `A` or `B`, we use a `Union` type, 
written `A | B`.

**Union and intersection types have not yet been implemented. Product 
types capture some of their use cases.**


# Composite operators

It is possible to define *composite* transformations: transformations 
that are derived from other, simpler ones. This should not necessarily 
be thought of as providing an *implementation*: it merely represents a 
decomposition into more primitive conceptual building blocks.

    >>> add = tf.Operator(type=Ratio ** Ratio ** Ratio)
    >>> add1 = tf.Operator(
            type=Ratio ** Ratio,
            define=lambda x: add(x, tf.Source(Ratio))
        )
    >>> compose = tf.Operation(
            type=lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            define=lambda f, g, x: f(g(x))
        )

When we use such composite operations, the underlying primitive 
expression can be derived using `Expr.primitive()`:

    >>> expr = lang.parse("compose add1 add1 -")
    >>> print(expr.primitive().tree())
    Ratio
     ├─Ratio ** Ratio
     │  ├─╼ add : Ratio ** Ratio ** Ratio
     │  └─Ratio
     │     ├─Ratio ** Ratio
     │     │  ├─╼ add : Ratio ** Ratio ** Ratio
     │     │  └─╼ - : Ratio
     │     └─╼ - : Ratio
     └─╼ - : Ratio

Composite expressions can be partially applied, which can further 
complicate the representation of [internal 
transformations](#internal-transformations). Furthermore, transformation 
graphs for composite expressions have not been fully implemented; see 
issue #40.


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
    operations, in which case the process is more involved. Consult the 
    chapter on [internal transformations](#internal-transformations).

These queries can themselves be represented as (partial) transformation 
graphs. We call such partial graphs *tasks*, because they are meant to 
express the conceptual properties we expect from a workflow that solves 
a particular task. For example, the following graph captures what we 
might want to see in a workflow that finds the closest hospital:

    @prefix : <https://github.com/quangis/transforge#>.
    @prefix stl: <https://example.com/stl/>.

    _:Hospitals stl:type "C(Obj)".
    _:Incident stl:type "Obj".
    [] a :Task;
        :input _:HospitalPoints, _:IncidentLocations;
        :output [
            stl:via "minimum";
            stl:type "Obj";
            :from [
                stl:via "distance";
                :from _:Hospitals;
                :from _:Incident
            ]
        ].

We would like to match this partial transformation graph to the 
*complete* transformation graph for a workflow. First, we need a SPARQL 
store containing workflows to match against. Assuming you have a SPARQL 
server like Fuseki or MarkLogic running, the command-line interface can 
help you upload your workflows:

    transforge graph -L sl.py -T wf.ttl wf.ttl \
        -s fuseki@http://127.0.0.1:3030/cct

You can once again use the command-line interface to query these 
workflows:

    transforge query -L sl.py task.ttl \
        -s fuseki@http://127.0.0.1:3030/cct -o -

This will add `:match` predicates to the task. With the `--summary` 
switch, a `.csv` file will be also be produced with an overview of all 
the matches.


# Canonical types

Compound types can be nested --- think `C(C(C(...)))`. Therefore, there 
exist infinite types. At the same time, the RDF graphs need to refer to 
these types by URI, and all their sub- and supertypes must be known so 
as to enable searching through them. For this reason, you can pass to 
your `Language` a set of *canonical types* via the `canon` parameter. 
This set of finite types, along with all their subtypes, is considered 
relevant. Non-canonical types will not be recorded into the 
transformation graph. For example:

    R = tf.TypeOperator(params=2)
    Qlt = tf.TypeOperator()
    Ord = tf.TypeOperator(supertype=Qlt)
    Ratio = tf.TypeOperator(supertype=Ord)

    example_lang = tf.Language(scope=locals(),
        canon={R(Qlt, Qlt)},
        namespace="https://example.com/stl/")

The `Top` and `Bottom` types may be included by adding them to the 
`canon`.

The command-line tool's `vocab` subcommand can generate an RDF 
vocabulary containing a description of all the operations of your 
language, as well as a *type taxonomy* of the canonical types. For the 
above language, it looks as follows:

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/quangis/transforge/develop/docs/resource/vocab.svg">
</p>

[cct]: https://github.com/quangis/cct
[wf]: http://geographicknowledge.de/vocab/Workflow.rdf
[sparql]: https://www.w3.org/TR/sparql11-query/
[w:ttl]: https://en.wikipedia.org/wiki/Turtle_(syntax)
[w:currying]: https://en.wikipedia.org/wiki/Currying
