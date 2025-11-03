---
name: memory-optimize
description: Memory usage optimization with garbage collection tuning and leak detection
category: optimization
version: 2.0.0
---

# 🧠 Memory Optimization

Optimize memory usage, detect leaks, and tune garbage collection.

```bash
node --max-old-space-size=4096 \
     --expose-gc \
     --trace-gc \
     server.js
```

**Version**: 2.0.0
