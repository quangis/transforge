import transformation_algebra as ct

Obj = ct.TypeOperator()
Ord = ct.TypeOperator()

size = ct.Operator(type=Obj ** Ord)
ratio = ct.Operator(type=Ord ** Ord ** Ord)

C = ct.TypeOperator(params=1)
height = ct.Operator(type=Obj ** Ord)
maximum = ct.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)

stl = ct.Language(scope=locals(), canon={Obj, Ord, C(Obj)},
    namespace="https://example.com/stl/")
