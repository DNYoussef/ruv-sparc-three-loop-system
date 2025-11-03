# Hive Mind Advanced - Gold Tier Skill

**Category**: Coordination | **Version**: 1.0.0 | **Tier**: Gold

## Quick Start

```bash
# Initialize hive mind with Byzantine consensus
npx claude-flow hive-mind init

# Spawn queen-led swarm
npx claude-flow hive-mind spawn "Build microservices" --queen-type strategic --consensus byzantine

# Monitor collective intelligence
npx claude-flow hive-mind status
```

## What You Get

### Core Capabilities
- **Queen-Led Coordination**: Strategic, tactical, and adaptive queen types
- **Byzantine Consensus**: 2/3 supermajority fault-tolerant decision making
- **Collective Memory**: Persistent shared knowledge with LRU caching
- **Auto-Scaling**: Dynamic worker spawning based on task load
- **Session Management**: Checkpoints, pause/resume, export/import

### Resources Included
- **Scripts**: 4 production-ready Python/JavaScript/Shell implementations
- **Templates**: 3 YAML/JSON configuration templates
- **Tests**: 3 comprehensive test suites with 90%+ coverage
- **Examples**: 3 real-world scenarios (150-300 lines each)

## File Structure

```
hive-mind-advanced/
├── skill.md              # Complete skill documentation
├── README.md             # This file
├── resources/
│   ├── scripts/
│   │   ├── queen-coordinator.py       # Strategic queen implementation
│   │   ├── consensus-manager.js       # Byzantine consensus engine
│   │   ├── collective-intelligence.sh # Shell orchestration
│   │   └── hive-optimizer.py          # Performance tuning
│   └── templates/
│       ├── hive-config.yaml           # Hive configuration template
│       ├── queen-protocol.json        # Queen behavior protocol
│       └── consensus-rules.yaml       # Consensus algorithm config
├── tests/
│   ├── test-queen-coordination.js     # Queen behavior tests
│   ├── test-consensus-byzantine.js    # Consensus algorithm tests
│   └── test-collective-memory.js      # Memory system tests
└── examples/
    ├── queen-led-swarm.js             # Full queen-worker coordination
    ├── byzantine-consensus.js         # Fault-tolerant decisions
    └── collective-decision-making.js  # Multi-agent voting
```

## Usage Patterns

### 1. Strategic Planning (Strategic Queen)
```bash
npx claude-flow hive-mind spawn "Research AI frameworks" \
  --queen-type strategic \
  --consensus weighted \
  --max-workers 6
```

### 2. Feature Implementation (Tactical Queen)
```bash
npx claude-flow hive-mind spawn "Build authentication API" \
  --queen-type tactical \
  --consensus majority \
  --max-workers 10
```

### 3. Performance Optimization (Adaptive Queen)
```bash
npx claude-flow hive-mind spawn "Optimize database queries" \
  --queen-type adaptive \
  --consensus byzantine \
  --max-workers 8
```

## Performance Benchmarks

- **Batch Spawning**: 10-20x faster with 5-agent batches
- **Overall Speed**: 2.8-4.4x improvement vs sequential
- **Token Reduction**: 32.3% through parallel coordination
- **Success Rate**: 84.8% on SWE-Bench tasks

## Integration Points

### With Claude Code
```bash
# Generate Claude Code spawn commands
npx claude-flow hive-mind spawn "Build full-stack app" --claude
```

### With SPARC Methodology
```bash
# Hive mind for TDD workflow
npx claude-flow sparc tdd "User auth" --hive-mind
```

### With GitHub
```bash
# PR review coordination
npx claude-flow hive-mind spawn "Review PR #123" --queen-type tactical
```

## Configuration

### Minimal Config
```yaml
objective: "Build microservices"
queenType: strategic
maxWorkers: 8
consensusAlgorithm: byzantine
```

### Advanced Config
```yaml
objective: "Full-stack development"
name: "fullstack-hive"
queenType: adaptive
maxWorkers: 12
consensusAlgorithm: weighted
autoScale: true
memorySize: 100  # MB
taskTimeout: 60  # minutes
encryption: true
```

## Testing

```bash
# Run all tests
cd tests
npm test

# Run specific test suite
npm test test-queen-coordination.js
npm test test-consensus-byzantine.js
npm test test-collective-memory.js
```

## Examples

### Run Examples
```bash
# Queen-led swarm coordination
node examples/queen-led-swarm.js

# Byzantine consensus demo
node examples/byzantine-consensus.js

# Collective decision making
node examples/collective-decision-making.js
```

## Troubleshooting

### High Memory Usage
```bash
npx claude-flow hive-mind memory --gc
npx claude-flow hive-mind memory --optimize
```

### No Consensus Reached
```bash
# Switch from byzantine to weighted
npx claude-flow hive-mind spawn "..." --consensus weighted
```

### Slow Task Assignment
```javascript
// Automatic worker type caching (built-in, no config needed)
// Caches best worker matches for 5 minutes
```

## Next Steps

1. **Read skill.md** for complete documentation
2. **Try examples/** for hands-on learning
3. **Run tests/** to verify your setup
4. **Use resources/templates/** for configuration
5. **Check resources/scripts/** for implementation patterns

## Related Skills

- `swarm-orchestration` - Basic swarm patterns
- `consensus-mechanisms` - Decision making algorithms
- `memory-systems` - Advanced memory management
- `sparc-methodology` - Structured development

## Support

- **Documentation**: See `skill.md` for full details
- **Issues**: https://github.com/ruvnet/claude-flow/issues
- **Community**: Claude Flow Discord

---

**Skill Tier**: Gold | **Last Updated**: 2025-11-02 | **Maintained By**: Claude Flow Team
