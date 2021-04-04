# Changelog

-   [breaking] Classes from the `type` module have been prefixed with `Type` 
    (bbc6eb9) or renamed entirely.
-   [breaking] Leaf expressions in transformation algebras are now defined in 
    terms of `Data` and `Operation` definitions, not just in terms of their 
    `Type`.
-   Leaf expressions can now be primitive or composite. Composite expressions 
    can be expanded into primitive expressions (bd766b7).
-   Unify singular constraints up to base types (6aa1487). This enables more 
    type inference.
-   Global module now imports user-facing objects from submodules (af0b587). 
    This makes the package more aerodynamic to use.

### v0.0.2

-   Type hints can now be found by [mypy](https://mypy.readthedocs.io/).

### v0.0.1

-   The type inference algorithm is mostly finished.
