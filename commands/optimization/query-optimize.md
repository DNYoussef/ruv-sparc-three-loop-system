---
name: query-optimize
description: Database query optimization with indexing and query analysis
category: optimization
version: 2.0.0
---

# 🗄️ Query Optimization

Optimize database queries with proper indexing and query analysis.

```sql
EXPLAIN ANALYZE
SELECT * FROM users WHERE status = 'active';

CREATE INDEX idx_users_status ON users(status);
```

**Version**: 2.0.0
