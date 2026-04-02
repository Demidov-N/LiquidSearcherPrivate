# Multi-Agent Worker System - Usage Guide

This project implements a three-tier worker system for OpenCode with progressive complexity escalation.

## Worker Tiers

### 🔧 @low-worker (Tier 1: Fast Execution)
- **Model**: Claude 3.5 Haiku (fastest, cheapest)
- **Scope**: 1-3 step tasks, simple operations
- **Use for**: File operations, formatting, templating, data extraction

```bash
# Examples
opencode "@low-worker Format all Python files in src/ with ruff"
opencode "@low-worker Read config.yaml and extract the database URL"
opencode "@low-worker Create a new React component from template"
```

### ⚙️ @mid-worker (Tier 2: Balanced Execution) 
- **Model**: Claude 3.5 Sonnet (balanced performance/cost)
- **Scope**: 4-8 step workflows, moderate complexity
- **Use for**: Feature implementation, refactoring, testing, debugging

```bash
# Examples  
opencode "@mid-worker Implement user authentication following this API spec"
opencode "@mid-worker Refactor the payment processing module"
opencode "@mid-worker Create comprehensive tests for the order system"
```

### 🧠 @high-worker (Tier 3: Strategic Execution)
- **Model**: Claude 3.5 Opus (maximum capability)
- **Scope**: 8+ steps, complex architecture and design
- **Use for**: System design, complex debugging, technology decisions

```bash
# Examples
opencode "@high-worker Design a scalable multi-tenant architecture"  
opencode "@high-worker Analyze this performance bottleneck and optimize"
opencode "@high-worker Plan migration from REST to GraphQL"
```

## Task Classification Rules

### Automatic Escalation
Workers automatically escalate when tasks exceed their complexity:
- **low-worker** → **mid-worker**: Multi-step reasoning needed
- **mid-worker** → **high-worker**: Architecture decisions required

### Manual Task Routing
Choose the appropriate worker based on:

| Factor | Low | Mid | High |
|--------|-----|-----|------|
| **Steps** | 1-3 | 4-8 | 8+ |
| **Files affected** | 1-2 | 3-10 | 10+ |
| **Reasoning depth** | None | Moderate | Deep |
| **Error impact** | Low | Medium | High |
| **Time estimate** | <2 min | 2-15 min | 15+ min |

## Delegation Patterns

### high-worker orchestration:
```
@high-worker "Build a user dashboard feature"
├─ Designs overall architecture
├─ @mid-worker: Implements React components  
├─ @mid-worker: Creates API endpoints
└─ @low-worker: Updates configuration files
```

### mid-worker delegation:
```
@mid-worker "Refactor authentication system"
├─ Plans refactoring approach
├─ @low-worker: Formats and cleans up files
├─ Implements core logic changes
└─ @low-worker: Updates import statements
```

## Best Practices

### 1. Start with the right tier
- **Simple tasks**: Start with @low-worker for speed
- **Complex features**: Start with @high-worker for planning
- **Implementation**: Use @mid-worker for execution

### 2. Let escalation work
- Don't force complex tasks on low-tier workers
- Trust the escalation system
- Workers know their limits

### 3. Use orchestration for large projects
```bash
# Good: Let high-worker orchestrate
opencode "@high-worker Build a complete authentication system"

# Less optimal: Manual coordination
opencode "@mid-worker Implement JWT service"
opencode "@mid-worker Create user model" 
opencode "@low-worker Update config files"
```

### 4. Monitor costs and performance
- **low-worker**: Cheapest, fastest for simple tasks
- **mid-worker**: Balanced cost/capability
- **high-worker**: Most expensive, use for strategic work only

## Configuration

Workers are configured in `opencode.json` with:
- **Model assignment**: Haiku/Sonnet/Opus per tier
- **Tool permissions**: Restricted access for safety
- **Step limits**: Bounded execution
- **Delegation rules**: Who can call whom

## Monitoring

Track usage patterns:
- Which tier handles most tasks?
- Are escalations working correctly?
- Cost per tier vs value delivered

The system learns and adapts based on success patterns.

---

**Quick Reference**:
- **@low-worker**: Files, formatting, templates (fast & cheap)
- **@mid-worker**: Features, refactoring, testing (balanced)  
- **@high-worker**: Architecture, strategy, complex problems (thorough)