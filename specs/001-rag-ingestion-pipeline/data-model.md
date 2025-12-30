# Data Model: RAG Ingestion Pipeline

## Entities

### BookContent
**Description**: Represents the text extracted from Docusaurus book URLs

**Fields**:
- `url`: string (required) - The source URL of the book content
- `title`: string (required) - The title of the page/document
- `content`: string (required) - The extracted text content
- `created_at`: datetime (required) - Timestamp when content was first ingested
- `updated_at`: datetime (required) - Timestamp when content was last updated
- `checksum`: string (required) - Hash of the content for change detection

**Validation Rules**:
- URL must be a valid HTTP/HTTPS URL
- Content must not be empty
- Checksum must be a valid SHA256 hash

### TextChunk
**Description**: A segment of book content processed according to the chunking strategy

**Fields**:
- `id`: string (required) - Stable, deterministic ID based on URL and position
- `content`: string (required) - The text content of the chunk (max 512 tokens)
- `source_url`: string (required) - Reference to the original BookContent URL
- `position`: integer (required) - Sequential position in the original document
- `overlap_prev`: string (optional) - Overlapping content from previous chunk
- `overlap_next`: string (optional) - Overlapping content from next chunk
- `token_count`: integer (required) - Number of tokens in the chunk
- `metadata`: object (required) - Additional metadata to store with the vector

**Validation Rules**:
- Content must be between 10 and 512 tokens
- Position must be non-negative
- Token count must match actual tokenization

### EmbeddingVector
**Description**: A numerical representation of a text chunk for semantic search

**Fields**:
- `chunk_id`: string (required) - Reference to the source TextChunk ID
- `vector`: array<float> (required) - The embedding vector from Cohere
- `model`: string (required) - The embedding model used to generate the vector
- `dimension`: integer (required) - The dimensionality of the vector
- `created_at`: datetime (required) - Timestamp when the embedding was generated

**Validation Rules**:
- Vector must have consistent dimensionality
- Model field must match the configured embedding model
- Dimension must match the expected Cohere model output

### QdrantRecord
**Description**: An entry in the Qdrant vector database containing an embedding vector and metadata

**Fields**:
- `id`: string (required) - The unique ID in Qdrant (same as TextChunk.id for consistency)
- `vector`: array<float> (required) - The embedding vector
- `payload`: object (required) - Metadata associated with the vector
  - `url`: string - Source URL
  - `title`: string - Document title
  - `position`: integer - Position in original document
  - `checksum`: string - Content checksum for change detection
  - `created_at`: datetime - When the record was created
  - `updated_at`: datetime - When the record was last updated
- `collection`: string (required) - The Qdrant collection name

**Validation Rules**:
- ID must be unique within the collection
- Vector dimension must match collection schema
- Payload must contain required metadata fields

## Relationships

1. **BookContent → TextChunk** (1 to many)
   - One BookContent document can be split into multiple TextChunk entities
   - Referenced by the `source_url` field in TextChunk

2. **TextChunk → EmbeddingVector** (1 to 1)
   - Each TextChunk has exactly one corresponding EmbeddingVector
   - Referenced by the `chunk_id` field in EmbeddingVector

3. **EmbeddingVector → QdrantRecord** (1 to 1)
   - Each EmbeddingVector is stored as one QdrantRecord
   - The QdrantRecord ID matches the TextChunk ID for consistency

## State Transitions

### BookContent
- `NEW` → `EXTRACTED` (when content is successfully extracted from URL)
- `EXTRACTED` → `CHANGED` (when content checksum differs from previous extraction)
- `CHANGED` → `EXTRACTED` (when updated content is re-extracted)

### TextChunk
- `CREATED` → `EMBEDDED` (when embedding is successfully generated)
- `EMBEDDED` → `STORED` (when successfully stored in Qdrant)

## Indexes

1. **BookContent.url**: For fast lookups by source URL
2. **TextChunk.source_url + TextChunk.position**: For ordered retrieval of chunks from a document
3. **QdrantRecord.id**: Primary key for vector lookups
4. **QdrantRecord.payload.url**: For finding all chunks from a specific URL