#!/bin/bash
# Simple script to query Weaviate collections

COLLECTION=${1:-Spanish}
LIMIT=${2:-5}

echo "Querying collection: $COLLECTION (limit: $LIMIT)"
echo "=========================================="
echo ""

curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"{ Get { ${COLLECTION}(limit: ${LIMIT}) { title text url _additional { id } } } }\"}" | python3 -m json.tool

