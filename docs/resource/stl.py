import transformation_algebra as ct

Obj = ct.TypeOperator()
Ord = ct.TypeOperator()

size = ct.Operator(type=Obj ** Ord)
ratio = ct.Operator(type=Ord ** Ord ** Ord)

stl = ct.Language(scope=locals(), canon={Obj, Ord},
    namespace="https://example.com/stl/")
