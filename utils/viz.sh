#!/bin/bash
# Convenience script to quickly visualize ttl files. Install dependencies:
#   sudo apt-get install xdot raptor2-utils # on Debian-based systems
#
# You may also be interested in <https://gephi.org/>. Note that you can also
# use RDFLib's `rdf2dot` to generate a `.dot` file in Python:
# >>> from rdflib.tools.rdf2dot import rdf2dot
# >>> with open("output.dot", 'w') as f:
# >>>     rdf2dot(graph, f)
# ... or on the command line:
# $ rdf2dot "$1" | xdot -

set -euo pipefail

if [ ! -z "${1:-}" ]; then
    rapper --input turtle "$1" \
    -f 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"' \
    -f 'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"' \
    -f 'xmlns:ta="https://github.com/quangis/transformation-algebra#"' \
    -f 'xmlns:cct="https://github.com/quangis/cct#"' \
    -f 'xmlns:ex="https://example.com/#"' \
    --output dot | xdot -
else
    echo "No input file." >& 2
fi
