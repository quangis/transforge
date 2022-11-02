# Changelog

### v0.2.0

This is a major update, with over 630 commits since the last version. 
The library's scope has changed. A lot of the original API still works, 
but since there's probably no users other than internal ones right now, 
I haven't put work into guaranteeing that all of it does. I'll give only 
a brief summary.

-   The library has been renamed to `transforge`.
-   It no longer accommodates only type inference and expression 
    parsing. Expressions can now also be converted into RDF graphs. As a 
    consequence, `rdflib` has become a dependency. Graphs for workflows, 
    in which each tool has an associated transformation expression, can 
    be converted into complex transformation graphs.
-   Transformation graphs can be converted into SPARQL queries to be 
    matched against other transformation graphs.
-   There is a command-line interface to facilitate creating 
    transformation graphs, uploading to graph stores, and querying from 
    those graph stores. As a consequence, `plumbum` has become a 
    dependency.

### v0.1.3

-   Partial primitives are now allowed in the final expression tree. They are 
    shown as lambda expressions and their function types are properly 
    resolved.
-   Variables are now printed with friendly names.

### v0.1.2

-   If a definition leads to error, the definition will be mentioned in the 
    error message.
-   Type inference has been improved: constraint options that are subtypes of 
    one another are boiled down to their most specific representative, 
    allowing more unification. Various other improvements to type inference.

### v0.1.1

Bugfix release. Fixes an infinite loop when constraining to a base type, and 
an issue with validation when an operation that is defined later is called.

### v0.1.0

Major changes:

-   [breaking] Base expressions in transformation algebras are now defined in 
    terms of `Data` and `Operation` definitions, not just in terms of their 
    `Type`.
-   Base expressions can now be primitive or composite. Composite expressions 
    can be expanded into primitive expressions (bd766b7). The inferred type 
    from composite expressions is checked against the declared type (c34008f).
-   Removed `pyparsing` dependency and sped up parsing by two orders of 
    magnitude. Python-style function calling (`f(x,y)`) is now parseable in 
    addition to `f x y`.
-   A constraint that has been narrowed down to a single remaining option will 
    now be now unified with its subject (except for base types). This enables 
    more type inference. See commit 6aa1487.
-   [breaking] There is no longer a distinction between constrained type 
    instances and plain type instances; constraints are directly attached to 
    variables. (4d5b741)

Minor changes:

-   [breaking] Directly pass globals to transformation algebra, if desired. 
    (6c12c34)
-   [breaking] Removed implicit conversion and application of types. See 
    commit messages at 7a854df and 8e212f6.
-   [breaking] Classes from the `type` module have been prefixed with `Type` 
    or renamed entirely (bbc6eb9).
-   Errors have been made clearer.
-   Constraints can now only contain variables that are present in the type 
    they are constraining (f3e7300).
-   Simplified the relationship between schematic types and type instances 
    (d697f8f).
-   The global module now imports user-facing objects from submodules 
    (af0b587). This makes the package more aerodynamic to use.

### v0.0.2

-   Type hints can now be found by [mypy](https://mypy.readthedocs.io/).

### v0.0.1

-   The type inference algorithm is mostly finished.
