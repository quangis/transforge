import transformation_algebra as ct

Obj = ct.TypeOperator()
Qlt = ct.TypeOperator()
Ord = ct.TypeOperator(supertype=Qlt)
Ratio = ct.TypeOperator(supertype=Ord)

size = ct.Operator(type=Obj ** Ord)
ratio = ct.Operator(type=Ord ** Ord ** Ord)

C = ct.TypeOperator(params=1)
height = ct.Operator(type=Obj ** Ord)
maximum = ct.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)

minimum = ct.Operator(type=lambda x: (x ** Ord) ** C(x) ** x)
distance = ct.Operator(type=Obj ** Obj ** Ratio)

stl = ct.Language(scope=locals(), canon={Obj, Qlt, C(Obj), C(Qlt)},
    namespace="https://example.com/stl/")
