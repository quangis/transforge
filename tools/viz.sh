#!/bin/bash
# Convenience script to quickly visualize ttl files. You may also be interested
# in <https://gephi.org/>.
#   Dependencies on Debian: sudo apt-get install xdot raptor2-utils
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
