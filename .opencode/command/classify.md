---
description: "Automatically classify task complexity and route to appropriate worker tier"
---

# Task Classification and Routing

Analyze the following task and determine the appropriate worker tier:

**Task**: {{task}}

## Classification Framework

Evaluate these factors:

### Complexity Indicators
- **Steps required**: How many distinct actions?
- **Files affected**: Single file or multiple?
- **Reasoning depth**: Pattern matching or deep analysis?
- **Error impact**: Low/medium/high risk?
- **Domain expertise**: Standard patterns or specialized knowledge?

### Decision Matrix

| Tier | Steps | Files | Reasoning | Examples |
|------|-------|-------|-----------|----------|
| **low-worker** | 1-3 | 1-2 | Pattern matching | File ops, formatting, templates |
| **mid-worker** | 4-8 | 3-10 | Multi-step logic | Features, refactoring, testing |
| **high-worker** | 8+ | 10+ | Deep analysis | Architecture, strategy, research |

## Analysis

Based on the task "{{task}}":

1. **Step Analysis**: 
2. **Scope Analysis**:
3. **Complexity Analysis**:
4. **Risk Analysis**:

## Recommendation

**Recommended tier**: [low-worker/mid-worker/high-worker]

**Reasoning**: 

**Suggested command**:
```bash
opencode "@[tier] {{task}}"
```

## Alternative Approaches

If this task could be broken down differently:
- **Option 1**: 
- **Option 2**:

---

*This classification helps optimize cost and performance by routing tasks to the most appropriate worker tier.*