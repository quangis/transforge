# Changelog

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
