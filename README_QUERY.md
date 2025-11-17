# Querying Weaviate Vector Database

This guide shows you how to view and query your embeddings in Weaviate.

## Quick Start

### 1. List All Collections
```bash
python query_weaviate.py --list
```

### 2. View Sample Data from a Collection
```bash
python query_weaviate.py --collection spanish --limit 10
```

### 3. View Data with Embeddings
```bash
python query_weaviate.py --collection spanish --limit 5 --show-embeddings
```

### 4. Count Objects in a Collection
```bash
python query_weaviate.py --collection spanish --count
```

## Accessing Weaviate via Web UI

### Option 1: Weaviate Console (Recommended)

Weaviate doesn't have a built-in web UI, but you can use the **Weaviate Console**:

1. **Install Weaviate Console** (if not already installed):
   ```bash
   npm install -g @weaviate/console
   ```

2. **Start the console**:
   ```bash
   weaviate-console
   ```
   
   This will open a web interface at `http://localhost:3000`

3. **Connect to your Weaviate instance**:
   - URL: `http://localhost:8080`
   - No authentication needed (if running locally)

### Option 2: Direct GraphQL API

You can query Weaviate directly using GraphQL:

```bash
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{
      Get {
        Spanish(limit: 5) {
          title
          text
          url
        }
      }
    }"
  }'
```

### Option 3: Python Script

Use the provided `query_weaviate.py` script:

```bash
# View first 10 objects
python query_weaviate.py --collection spanish --limit 10

# View with embeddings
python query_weaviate.py --collection spanish --limit 3 --show-embeddings

# Count total objects
python query_weaviate.py --collection spanish --count
```

## Example Queries

### View Sample Documents
```bash
python query_weaviate.py --collection spanish --limit 5
```

### View Embedding Vectors
```bash
python query_weaviate.py --collection spanish --limit 2 --show-embeddings
```

Output will show:
- Object UUID
- All properties (title, text, url, text_raw)
- Vector dimensions
- First and last 10 values of the embedding
- Min/Max values

### List All Collections
```bash
python query_weaviate.py --list
```

## Using Weaviate Console (Web UI)

1. **Install**:
   ```bash
   npm install -g @weaviate/console
   ```

2. **Run**:
   ```bash
   weaviate-console
   ```

3. **Access**: Open `http://localhost:3000` in your browser

4. **Connect**: Enter `http://localhost:8080` as the Weaviate URL

5. **Query**: Use the GraphQL query interface to explore your data

## GraphQL Query Examples

### Get all objects from a collection:
```graphql
{
  Get {
    Spanish(limit: 10) {
      title
      text
      url
      _additional {
        id
      }
    }
  }
}
```

### Get objects with vectors:
```graphql
{
  Get {
    Spanish(limit: 5) {
      title
      text
      _additional {
        id
        vector
      }
    }
  }
}
```

### Vector Search (semantic search):
```graphql
{
  Get {
    Spanish(
      nearText: {
        concepts: ["machine learning"]
      }
      limit: 5
    ) {
      title
      text
      _additional {
        distance
      }
    }
  }
}
```

## Notes

- **Embeddings are large**: Each embedding vector has hundreds of dimensions (typically 768 or more)
- **Performance**: Showing embeddings for many objects can be slow
- **Storage**: Embeddings are stored as vectors in Weaviate and used for similarity search
- **Query Speed**: Vector searches are fast even with millions of vectors

## Troubleshooting

### Connection Issues
- Make sure Weaviate is running: `docker ps`
- Check the URL: `curl http://localhost:8080/v1/meta`

### Collection Not Found
- List collections: `python query_weaviate.py --list`
- Check collection name (case-sensitive)

### No Data
- Wait for insertion to complete
- Check insertion progress in the terminal

