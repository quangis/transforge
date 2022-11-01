import transformation_algebra as ct

R = ct.TypeOperator(params=2)
Qlt = ct.TypeOperator()
Ord = ct.TypeOperator(supertype=Qlt)
Ratio = ct.TypeOperator(supertype=Ord)

lang = ct.Language(scope=locals(),
    canon={R(Qlt, Qlt)},
    namespace="https://example.com/stl/")
