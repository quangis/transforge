# Graphs and queries

Beyond *expressing* transformations, an additional goal of the library is to 
enable *querying* them for their constituent operations and data types.

To turn an expression into a searchable structure, we convert it to an RDF 
graph. Every data source and every operation applied to it becomes a node, 
representing the type of data that is conceptualized at that particular step in 
the transformation. Chains of nodes are thus obtained that are easily subjected 
to queries along the lines of: 'find me a transformation containing operations 
`f` and `g` that, somewhere downstream, combine into data of type `t`'.

The process is straightforward when operations only take data as input. 
However, expressions in an algebra may also take other operations, in which 
case the process is more involved; for now, consult the source code.

In practical terms, to obtain a graph representation of the previous 
expression, you may do:

    >>> g = ta.TransformationGraph()
    >>> g.add_expr(expr)
    >>> g.serialize("graph.ttl", format="ttl")

These graphs can be queried via constructs from the [SPARQL 1.1 
specification](https://www.w3.org/TR/sparql11-query/).

