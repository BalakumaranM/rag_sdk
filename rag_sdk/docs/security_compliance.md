# Security & Compliance {#security-compliance}

## Multi-Tenancy

The SDK is built for multi-tenant environments with strict data isolation.

### Isolation Strategies

1.  **Namespace Isolation (Recommended):** Uses vector store namespaces (e.g., `tenant_123`) to logically separate data.
2.  **Separate Index:** Physically separate indices for maximum isolation.
3.  **Encryption:** Tenant-specific encryption keys.

```python
from rag_sdk.multitenancy import TenantContext

# Automatic tenant isolation
with TenantContext(tenant_id="company_a"):
    rag.ingest_documents(documents)  # Automatically namespaced
    response = rag.query("query")  # Can only access company_a data
```

## PII Protection

We provide a multi-layer PII protection system:

1.  **Detection:** Identifying sensitive entities (SSN, credit cards, emails) using Presidio or AWS Comprehend.
2.  **Handling:** Redaction, encryption, or exclusion.

```yaml
pii_protection:
  detection:
    enabled: true
    entities: ["EMAIL", "SSN", "CREDIT_CARD"]
  handling:
    strategy: "redact"
    redact:
      replacement: "[REDACTED_{entity_type}]"
```

## Prompt Injection Safeguards

To prevent malicious prompt injection attacks, the SDK includes:

*   **Input Validation:** Detecting injection patterns ("ignore previous instructions").
*   **Sandboxing:** Using signed system prompts and guardrails.
*   **Output Validation:** Checking for instruction leakage in responses.

## GDPR & Data Residency

### Data Residency
Configure region-specific storage and processing to comply with local laws.

```yaml
compliance:
  data_residency:
    regions:
      eu:
        vectorstore: { environment: "eu-west1-gcp" }
        llm: { endpoint: "https://eu-openai.azure.com" }
```

### Right to be Forgotten
The SDK supports efficient data deletion across all stores (vector, metadata, cache) to comply with GDPR deletion requests.

```python
rag.handle_deletion_request(user_id="user_123", request_id="req_001")
```

## Audit Logs

Comprehensive audit logging for all sensitive operations:
*   Document ingestion/deletion
*   Query execution
*   PII detection/access
*   Configuration changes

Logs can be stored in immutable storage (e.g., S3 Object Lock) for compliance. 
