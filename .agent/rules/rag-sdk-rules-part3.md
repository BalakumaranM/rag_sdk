# RAG SDK Development Rules - Part 3: Production & Operations

## 🔒 Security & Privacy Requirements

### Must-Have Features
1. **PII Detection & Redaction**
   - Detect email, phone, SSN, credit cards
   - Configurable redaction strategies
   - Audit trail of redacted content

2. **Multi-Tenancy**
   - Namespace isolation in vector stores
   - Tenant-specific encryption keys
   - Query isolation and access control

3. **Prompt Injection Defense**
   - Input sanitization
   - Delimiter-based isolation
   - Content filtering

4. **Encryption**
   - At-rest: AES-256
   - In-transit: TLS 1.3
   - Key rotation support

5. **Compliance**
   - GDPR: Right to deletion, data export
   - HIPAA: Audit logging, access controls
   - SOC 2: Comprehensive audit trails

---

## ⚡ Performance & Scalability Standards

### Performance Targets
- Query latency: < 500ms (p95)
- Indexing throughput: > 1000 docs/sec
- Concurrent queries: Support 100+ simultaneous
- Memory efficiency: < 2GB for 1M documents (with FAISS)

### Scalability Patterns
- **Horizontal scaling**: Stateless design, load balancing support
- **Caching**: Multi-layer (in-memory, Redis, CDN)
- **Batch processing**: For embeddings and indexing
- **Rate limiting**: Token bucket algorithm
- **Connection pooling**: For all external services
- **Lazy loading**: Only load components when needed

### Async Everything
```python
# Good
async def process_document(self, doc: Document) -> ProcessedDocument:
    chunks = await self.splitter.split(doc)
    embeddings = await self.embedder.embed_batch(chunks)
    await self.vectorstore.upsert(chunks, embeddings)
    return ProcessedDocument(chunks=chunks)

# Bad - synchronous blocking
def process_document(self, doc: Document) -> ProcessedDocument:
    chunks = self.splitter.split(doc)
    embeddings = self.embedder.embed_batch(chunks)  # Blocks!
    self.vectorstore.upsert(chunks, embeddings)
    return ProcessedDocument(chunks=chunks)
```

---

## 📊 Observability Requirements

### Logging Levels
```python
import structlog

logger = structlog.get_logger(__name__)

# DEBUG: Detailed diagnostic info
logger.debug("chunk_size", size=len(chunk), chunk_id=chunk.id)

# INFO: Important events
logger.info("document_processed", doc_id=doc.id, chunks=len(chunks))

# WARNING: Recoverable issues
logger.warning("rate_limit_hit", provider="openai", retry_in=30)

# ERROR: Failed operations
logger.error("embedding_failed", doc_id=doc.id, error=str(e))
```

### Metrics to Track
- Request counts by provider
- Latency percentiles (p50, p95, p99)
- Error rates
- Cache hit rates
- Token usage
- Vector store operations

### Tracing
Use OpenTelemetry for distributed tracing:
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def query(self, query: str):
    with tracer.start_as_current_span("rag.query") as span:
        span.set_attribute("query.length", len(query))
        
        # Embed query
        with tracer.start_as_current_span("embed.query"):
            embedding = await self.embedder.embed_query(query)
        
        # Retrieve
        with tracer.start_as_current_span("vectorstore.search"):
            results = await self.vectorstore.search(embedding)
        
        # Generate
        with tracer.start_as_current_span("llm.generate"):
            response = await self.llm.generate(query, results)
        
        return response
```

---

## 🚨 Common Pitfalls to Avoid

### 1. Synchronous I/O in Async Functions
```python
# ❌ Bad
async def embed_documents(self, texts):
    return requests.post(url, json=texts)  # Blocking!

# ✅ Good
async def embed_documents(self, texts):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=texts) as resp:
            return await resp.json()
```

### 2. Missing Error Context
```python
# ❌ Bad
except Exception as e:
    raise Exception("Failed")

# ✅ Good
except RateLimitError as e:
    raise RateLimitError(
        f"OpenAI rate limit exceeded for model {self.model}. "
        f"Retry after {e.retry_after} seconds. "
        f"Original error: {e}"
    ) from e
```

### 3. Hardcoded Configuration
```python
# ❌ Bad
CHUNK_SIZE = 1000

# ✅ Good
self.chunk_size = config.get("chunk_size", 1000)
```

### 4. No Input Validation
```python
# ❌ Bad
def split_text(self, text):
    return text.split()

# ✅ Good
def split_text(self, text: str) -> List[str]:
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not text.strip():
        raise ValueError("Text cannot be empty")
    return text.split()
```

### 5. Leaking Resources
```python
# ❌ Bad
def load_file(self, path):
    f = open(path)
    return f.read()

# ✅ Good
def load_file(self, path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
```

### 6. Not Handling Rate Limits
```python
# ❌ Bad
async def embed(self, text):
    return await api.embed(text)  # Will fail on rate limit

# ✅ Good
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def embed(self, text):
    try:
        return await api.embed(text)
    except RateLimitError as e:
        logger.warning(f"Rate limit hit, retrying in {e.retry_after}s")
        raise
```

### 7. Ignoring Context Window Limits
```python
# ❌ Bad
def build_prompt(self, query, docs):
    context = "\n\n".join([d.content for d in docs])
    return f"Context: {context}\n\nQuery: {query}"

# ✅ Good
def build_prompt(self, query, docs, max_tokens=4000):
    context_parts = []
    token_count = len(self.tokenizer.encode(query))
    
    for doc in docs:
        doc_tokens = len(self.tokenizer.encode(doc.content))
        if token_count + doc_tokens > max_tokens:
            break
        context_parts.append(doc.content)
        token_count += doc_tokens
    
    context = "\n\n".join(context_parts)
    return f"Context: {context}\n\nQuery: {query}"
```

---

## 📈 Success Metrics

The SDK will be considered world-class when it achieves:

### Technical Metrics
- Test coverage > 90%
- Type coverage > 95% (mypy strict)
- Zero critical security vulnerabilities
- API response time < 500ms (p95)
- Documentation coverage: 100% of public API

### User Experience Metrics
- Time to "Hello World": < 5 minutes
- Time to production deployment: < 1 day
- GitHub stars (target: 1000+ in 6 months)
- Community contributions (PRs, issues)

### Code Quality Metrics
- Maintainability index: A grade
- Cyclomatic complexity: < 10 per function
- Code duplication: < 3%
- Documentation quality score: > 90%

---

## 🎯 Production Readiness Checklist

### Before v1.0 Release
- [ ] All core features implemented and tested
- [ ] Security audit completed
- [ ] Performance benchmarks meet targets
- [ ] Documentation complete (README, tutorials, API docs)
- [ ] Examples for all major use cases
- [ ] Error messages are helpful and actionable
- [ ] Logging is comprehensive but not noisy
- [ ] Metrics and monitoring in place
- [ ] License file included (MIT or Apache 2.0)
- [ ] Contributing guidelines published
- [ ] Code of conduct added
- [ ] CI/CD pipeline configured
- [ ] PyPI package published
- [ ] Docker images available
- [ ] Helm charts for Kubernetes (optional but nice)

### Deployment Checklist
- [ ] Environment variables documented
- [ ] Configuration validation on startup
- [ ] Health check endpoints implemented
- [ ] Graceful shutdown handling
- [ ] Resource limits configured
- [ ] Backup and restore procedures documented
- [ ] Disaster recovery plan in place
- [ ] Monitoring dashboards created
- [ ] Alerting rules configured
- [ ] Runbooks for common issues

---

## 💡 Best Practices Summary

### Do's ✅
- Use type hints everywhere
- Write async code for I/O operations
- Validate all inputs
- Provide meaningful error messages
- Log important events with context
- Test edge cases and error paths
- Document public APIs thoroughly
- Use design patterns appropriately
- Handle rate limits gracefully
- Implement retry logic with backoff
- Cache expensive operations
- Monitor performance metrics
- Secure sensitive data
- Follow PEP 8 style guide
- Keep functions small and focused

### Don'ts ❌
- Don't block the event loop
- Don't hardcode configuration
- Don't ignore errors silently
- Don't skip input validation
- Don't leak resources (files, connections)
- Don't use print() for logging
- Don't make assumptions about user data
- Don't expose internal implementation details
- Don't break backward compatibility without major version bump
- Don't sacrifice security for convenience
- Don't skip documentation
- Don't write tests after the code
- Don't optimize prematurely
- Don't reinvent the wheel (use well-tested libraries)
- Don't commit secrets to version control

---

## 🎉 Final Notes

This is an ambitious project to build a **production-grade, enterprise-ready RAG SDK**. Quality over speed. Understanding over copying. Learning over finishing quickly.

**Remember:**
- Every line of code should be understood, not just working
- Every component should be tested
- Every decision should be documented
- Every API should be intuitive

**Success looks like:**
- A developer can integrate the SDK in minutes
- The code is maintainable by others
- The system scales to production workloads
- The user learned valuable skills building it

**When in doubt:**
- Ask questions
- Seek clarification
- Propose alternatives
- Explain trade-offs

Let's build something world-class! 🚀

---

**← See Part 1 for Foundation & Architecture | See Part 2 for Implementation & Quality**
