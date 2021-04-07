# Changelog

-   [breaking] Removed implicit conversion and application of types. See 
    commit messages at 7a854df and 8e212f6.
-   [breaking] Classes from the `type` module have been prefixed with `Type` 
    or renamed entirely (bbc6eb9).
-   [breaking] Base expressions in transformation algebras are now defined in 
    terms of `Data` and `Operation` definitions, not just in terms of their 
    `Type`.
-   [breaking] There is no longer a distinction between constrained type 
    instances and plain type instances; constraints are directly attached to 
    variables. (4d5b741)
-   Constraints can now only contain variables that are present in the type 
    they are constraining (f3e7300).
-   Simplified the relationship between schematic types and type instances 
    (d697f8f).
-   Base expressions can now be primitive or composite. Composite expressions 
    can be expanded into primitive expressions (bd766b7). The inferred type 
    from composite expressions is checked against the declared type (c34008f).
-   A constraint with a single remaining option will now be now unified until 
    its base type (6aa1487). This enables more type inference.
-   The global module now imports user-facing objects from submodules 
    (af0b587). This makes the package more aerodynamic to use.

### v0.0.2

-   Type hints can now be found by [mypy](https://mypy.readthedocs.io/).

### v0.0.1

-   The type inference algorithm is mostly finished.
