# RAG SDK Development Rules - Part 2: Implementation & Quality

## 🎨 API Design Principles

### 1. Consistency
All providers should have identical interfaces:
```python
# Good - consistent interface
embedder = OpenAIEmbeddings(config)
result = await embedder.embed_query("text")

embedder = CohereEmbeddings(config)
result = await embedder.embed_query("text")  # Same method!

# Bad - inconsistent
result = await openai.get_embedding("text")
result = await cohere.create_embedding("text")  # Different!
```

### 2. Fail Fast
Validate inputs immediately:
```python
def __init__(self, api_key: str):
    if not api_key:
        raise ValueError("API key is required")
    if not self._validate_key(api_key):
        raise ValueError("Invalid API key format")
    self.api_key = api_key
```

### 3. Meaningful Errors
```python
# Good
raise VectorStoreError(
    f"Failed to connect to Pinecone index '{index_name}'. "
    f"Verify the index exists in environment '{environment}' "
    f"and your API key has access. "
    f"Error: {e}"
)

# Bad
raise Exception("Connection failed")
```

### 4. Sensible Defaults
```python
class RetrieverConfig:
    top_k: int = 5  # Good default
    score_threshold: float = 0.7
    enable_reranking: bool = False  # Safe default
```

---

## 🧪 Testing Requirements

### Test Pyramid
1. **Unit Tests (70%)**
   - Test individual functions and classes
   - Mock external dependencies
   - Fast execution (< 1 second total)

2. **Integration Tests (20%)**
   - Test component interactions
   - Use test doubles for external services
   - Moderate execution time (< 30 seconds total)

3. **End-to-End Tests (10%)**
   - Full pipeline tests
   - May use real services (with test data)
   - Longer execution time acceptable

### Test Coverage Requirements
```python
# Example: Test ALL code paths
@pytest.mark.parametrize("provider,expected", [
    ("openai", OpenAIEmbeddings),
    ("cohere", CohereEmbeddings),
    ("huggingface", HuggingFaceEmbeddings),
])
def test_embedding_factory(provider, expected):
    config = {"api_key": "test-key"}
    embedder = EmbeddingFactory.create(provider, config)
    assert isinstance(embedder, expected)

def test_embedding_factory_invalid_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        EmbeddingFactory.create("invalid", {})
```

### Required Test Types
- Unit tests with pytest
- Property-based testing with Hypothesis
- Async tests with pytest-asyncio
- Mock external APIs with responses or aioresponses
- Test fixtures for common scenarios
- Performance benchmarks

---

## 📚 Documentation Standards

### Code Documentation
Every public function/class must have:
```python
async def embed_documents(
    self,
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """Embed multiple documents using the configured model.
    
    This method batches requests to optimize API usage and handles
    rate limiting automatically.
    
    Args:
        texts: List of text strings to embed. Each should be < 8191 tokens
               for OpenAI models.
        batch_size: Number of texts to embed in each API call. Defaults to
                   100 to optimize throughput while respecting rate limits.
    
    Returns:
        List of embedding vectors, one per input text. Each vector has
        dimensionality determined by the model (e.g., 1536 for
        text-embedding-3-small).
    
    Raises:
        ValueError: If any text is empty or exceeds token limit.
        RateLimitError: If API rate limit is exceeded after retries.
        EmbeddingError: If embedding generation fails.
    
    Example:
        >>> embedder = OpenAIEmbeddings(api_key="sk-...")
        >>> texts = ["Hello world", "Goodbye world"]
        >>> embeddings = await embedder.embed_documents(texts)
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        1536
    
    Note:
        This method automatically retries failed requests up to 3 times
        with exponential backoff.
    """
```

### User Documentation
- **README**: Quick start, installation, basic usage
- **Tutorials**: Step-by-step guides for common scenarios
- **API Reference**: Auto-generated from docstrings
- **Architecture Guide**: System design and component interactions
- **Migration Guides**: Version upgrade instructions
- **Troubleshooting**: Common issues and solutions

---

## 🚀 Development Workflow

### When Implementing a New Feature

1. **Understand Requirements**
   - Ask clarifying questions
   - Discuss trade-offs
   - Propose architecture

2. **Design API**
   - Sketch interface first
   - Get feedback before implementing
   - Consider backward compatibility

3. **Implement with TDD**
   - Write tests first
   - Implement minimal code to pass
   - Refactor for clarity

4. **Document**
   - Add docstrings
   - Update user documentation
   - Create examples

5. **Review**
   - Self-review checklist
   - Test edge cases
   - Benchmark performance

### Code Review Checklist
- [ ] Type hints on all functions
- [ ] Docstrings with examples
- [ ] Error handling for failure modes
- [ ] Tests covering happy path and edge cases
- [ ] No hardcoded values (use config)
- [ ] Async for I/O operations
- [ ] Proper logging (no print statements)
- [ ] Security considerations addressed
- [ ] Performance acceptable
- [ ] Documentation updated

---

## 🔍 Code Quality Checklist

Before accepting any code, verify:

### Functionality
- [ ] Solves the stated problem correctly
- [ ] Handles edge cases
- [ ] Graceful error handling
- [ ] Input validation

### Code Quality
- [ ] Follows Python PEP 8 style guide
- [ ] Type hints on all functions
- [ ] Clear, descriptive variable names
- [ ] No code duplication (DRY principle)
- [ ] Functions are focused (single responsibility)
- [ ] Complexity is reasonable (avoid nested loops/conditions)

### Testing
- [ ] Unit tests written
- [ ] Tests are readable and maintainable
- [ ] Edge cases tested
- [ ] Mocking used appropriately
- [ ] Test coverage > 80%

### Documentation
- [ ] Docstrings on all public functions
- [ ] Complex logic has inline comments
- [ ] User-facing docs updated
- [ ] Example usage provided

### Performance
- [ ] No obvious performance issues
- [ ] Async used for I/O
- [ ] Resources properly cleaned up
- [ ] Memory usage reasonable

### Security
- [ ] No hardcoded secrets
- [ ] Input sanitized
- [ ] SQL injection prevention (if applicable)
- [ ] XSS prevention (if applicable)

---

## 🎓 Learning-Focused Interaction Guidelines

### For the AI Agent

1. **Explain Before Coding**
   ```
   Before I provide the code, let me explain the approach:
   
   We're implementing the Strategy pattern here because we need to support
   multiple embedding providers (OpenAI, Cohere, etc.) with the same interface.
   
   Key components:
   1. Abstract base class (EmbeddingProvider)
   2. Concrete implementations (OpenAIEmbeddings, etc.)
   3. Factory for creating instances
   
   This allows users to swap providers by just changing config, without
   code changes. Does this make sense?
   ```

2. **Progressive Complexity**
   - Start with simple, working version
   - Explain limitations
   - Progressively add features
   - Explain each enhancement

3. **Ask Questions**
   ```
   Before implementing the caching layer, I need to understand your
   requirements better:
   
   1. What's your expected cache size? (affects cache strategy)
   2. Do you need distributed caching? (Redis vs in-memory)
   3. What's your cache invalidation strategy? (TTL, LRU, etc.)
   ```

4. **Provide Alternatives**
   ```
   We have three options for text splitting:
   
   Option 1: Fixed-size chunks (simple, predictable, may split sentences)
   Option 2: Recursive (respects sentence boundaries, variable size)
   Option 3: Semantic (best quality, slowest, needs embedding model)
   
   For a production SDK, I recommend implementing all three and letting
   users choose via config. Want to start with recursive as default?
   ```

5. **Review Together**
   After providing code:
   ```
   Let's review what we just implemented:
   
   1. What does the @retry decorator do?
   2. Why did we use asyncio.gather() here?
   3. What edge case does line 47 handle?
   
   Understanding these will help when you extend this code later.
   ```

---

## 🤝 Interaction Protocol

### When User Asks for Code
1. **Understand**: "Before I code this, let me make sure I understand..."
2. **Design**: "Here's how I'd approach this architecturally..."
3. **Explain**: "The key concepts we'll use are..."
4. **Implement**: Provide code with inline comments
5. **Review**: "Let's walk through what this code does..."
6. **Test**: "Here's how we'd test this..."
7. **Document**: "And here's the documentation..."

### When User Asks "How Do I..."
1. Explain the concept
2. Show the SDK way to do it
3. Provide complete example
4. Explain why this approach is best
5. Mention alternatives if any

### When You're Unsure
- Ask clarifying questions
- Don't make assumptions
- Propose multiple options
- Explain trade-offs

---

**← See Part 1 for Foundation & Architecture | Continue to Part 3 for Production & Operations →**
