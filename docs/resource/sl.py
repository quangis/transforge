import transformation_algebra as ct

Obj = ct.TypeOperator()
Ord = ct.TypeOperator()
C = ct.TypeOperator(params=1)

distance = ct.Operator(type=Obj ** Obj ** Ord)
minimum = ct.Operator(type=(Obj ** Ord) ** C(Obj) ** Obj)

sl = ct.Language(scope=locals(), canon={Obj, Ord, C(Obj)})
