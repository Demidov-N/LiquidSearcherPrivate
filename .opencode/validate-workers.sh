#!/bin/bash
# Validate the three-tier worker system setup

echo "🔍 Validating OpenCode Three-Tier Worker System..."
echo ""

# Check configuration file
if [ -f "opencode.json" ]; then
    echo "✅ opencode.json exists"
    
    # Check if agents are defined
    if grep -q "low-worker" opencode.json; then
        echo "✅ low-worker configured"
    else
        echo "❌ low-worker missing"
    fi
    
    if grep -q "mid-worker" opencode.json; then
        echo "✅ mid-worker configured"
    else
        echo "❌ mid-worker missing"
    fi
    
    if grep -q "high-worker" opencode.json; then
        echo "✅ high-worker configured"
    else
        echo "❌ high-worker missing"
    fi
else
    echo "❌ opencode.json not found"
fi

echo ""

# Check agent files
if [ -f ".opencode/agent/low-worker.md" ]; then
    echo "✅ low-worker.md exists"
else
    echo "❌ low-worker.md missing"
fi

if [ -f ".opencode/agent/mid-worker.md" ]; then
    echo "✅ mid-worker.md exists"
else
    echo "❌ mid-worker.md missing"
fi

if [ -f ".opencode/agent/high-worker.md" ]; then
    echo "✅ high-worker.md exists"
else
    echo "❌ high-worker.md missing"
fi

echo ""

# Check documentation
if [ -f ".opencode/WORKER_SYSTEM.md" ]; then
    echo "✅ Usage guide exists"
else
    echo "❌ Usage guide missing"
fi

if [ -f ".opencode/command/classify.md" ]; then
    echo "✅ Classification command exists"
else
    echo "❌ Classification command missing"
fi

echo ""
echo "🎯 Testing worker invocation (if OpenCode is available):"

if command -v opencode &> /dev/null; then
    echo "✅ OpenCode CLI available"
    echo ""
    echo "Test commands:"
    echo "  opencode '/classify task=\"Format Python files\"'"
    echo "  opencode '@low-worker Read the README.md file'"
    echo "  opencode '@mid-worker Create a simple test file'"
    echo "  opencode '@high-worker Analyze the project structure'"
else
    echo "⚠️  OpenCode CLI not in PATH (install and configure first)"
fi

echo ""
echo "📁 Directory structure:"
find .opencode -type f -name "*.md" | sort
echo ""
echo "✨ Three-tier worker system validation complete!"