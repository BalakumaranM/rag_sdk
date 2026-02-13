# PostgreSQL Quick Reference

## Core Rules
- **Primary Key**: `BIGINT GENERATED ALWAYS AS IDENTITY` preferred; `UUID` for distributed systems
- **Normalize first** (3NF), denormalize only for proven performance needs
- **NOT NULL** everywhere semantically required
- **Always index FK columns** (PostgreSQL doesn't auto-create these!)

## Data Types

| Type | Use For |
|------|---------|
| `BIGINT GENERATED ALWAYS AS IDENTITY` | Primary keys |
| `TEXT` | Strings (prefer over VARCHAR) |
| `TIMESTAMPTZ` | Timestamps (not TIMESTAMP) |
| `NUMERIC(p,s)` | Money (never float) |
| `BOOLEAN NOT NULL` | True/false |
| `JSONB` | Semi-structured data |
| `UUID` | Distributed/opaque IDs |

### Avoid These
- `timestamp` → use `timestamptz`
- `char(n)` / `varchar(n)` → use `text`
- `money` type → use `numeric`
- `serial` → use `generated always as identity`

## Index Types

| Type | Use Case | Example |
|------|----------|---------|
| B-tree | Equality, range, ORDER BY | Default |
| GIN | JSONB, arrays, full-text | `USING GIN (attrs)` |
| GiST | Ranges, geometry, exclusion | `USING GiST (period)` |
| BRIN | Time-series, naturally ordered | `USING BRIN (created_at)` |

## Common Patterns

```sql
-- Users table
CREATE TABLE users (
  user_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX ON users (LOWER(email));

-- Orders with FK (always add index!)
CREATE TABLE orders (
  order_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(user_id),
  status TEXT NOT NULL DEFAULT 'PENDING' 
    CHECK (status IN ('PENDING', 'PAID', 'CANCELED')),
  total NUMERIC(10,2) NOT NULL CHECK (total > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON orders (user_id);  -- REQUIRED for FK!
CREATE INDEX ON orders (created_at);

-- JSONB profile
CREATE TABLE profiles (
  user_id BIGINT PRIMARY KEY REFERENCES users(user_id),
  attrs JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX ON profiles USING GIN (attrs);
```

## JSONB Queries
```sql
-- Containment (uses GIN index)
SELECT * FROM profiles WHERE attrs @> '{"theme": "dark"}';

-- Key existence
SELECT * FROM profiles WHERE attrs ? 'preferences';

-- Extract value
SELECT attrs->>'theme' FROM profiles;
```

## Gotchas
- **Unquoted identifiers are lowercased** → use snake_case
- **UNIQUE allows multiple NULLs** → use `NULLS NOT DISTINCT` (PG15+)
- **FK columns need explicit indexes** → PostgreSQL doesn't create these
- **Sequences have gaps** → normal behavior, don't "fix"
