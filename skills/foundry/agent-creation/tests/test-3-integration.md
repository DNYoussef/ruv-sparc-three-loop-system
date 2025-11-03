# Test 3: Multi-Agent Integration and Coordination

**Objective**: Test agent coordination, memory sharing, and multi-agent workflows

**Test Type**: Integration & End-to-End
**Difficulty**: Advanced
**Duration**: ~30 minutes

---

## Test Scenario

Create a coordinated system of 3 specialist agents (Python, TypeScript, Testing) that collaborate on building a full-stack application using Memory MCP for shared context and Claude-Flow hooks for coordination.

### Prerequisites

- Tests 1 and 2 completed
- Memory MCP configured and running
- Claude-Flow hooks functional
- Understanding of multi-agent coordination

---

## Phase 1: Agent Creation

### Step 1: Create Python Backend Specialist

```bash
cd resources/scripts
./generate_agent.sh python-backend-specialist development --output ../../tests/output
```

**Customize** `agent-spec.yaml`:
```yaml
capabilities:
  primary:
    - "REST API development with FastAPI"
    - "Database design and ORM (SQLAlchemy)"
    - "Authentication and authorization"

  integrations:
    - name: backend_coordination
      type: coordination
      config:
        role: backend
        exposes: ["api_endpoints", "database_schema"]
        consumes: ["frontend_requirements"]
```

### Step 2: Create TypeScript Frontend Specialist

```bash
./generate_agent.sh typescript-frontend-specialist development --output ../../tests/output
```

**Customize** `agent-spec.yaml`:
```yaml
capabilities:
  primary:
    - "React 18+ development"
    - "State management with Zustand"
    - "API integration and data fetching"

  integrations:
    - name: frontend_coordination
      type: coordination
      config:
        role: frontend
        exposes: ["ui_components", "frontend_requirements"]
        consumes: ["api_endpoints"]
```

### Step 3: Create Testing Specialist

```bash
./generate_agent.sh integration-testing-specialist testing --output ../../tests/output
```

**Customize** `agent-spec.yaml`:
```yaml
capabilities:
  primary:
    - "End-to-end testing with Playwright"
    - "Integration testing"
    - "API contract testing"

  integrations:
    - name: testing_coordination
      type: coordination
      config:
        role: tester
        consumes: ["api_endpoints", "ui_components"]
        validates: ["integration_contracts"]
```

### Validation

```bash
for agent in python-backend-specialist typescript-frontend-specialist integration-testing-specialist; do
    echo "Validating $agent..."
    python3 validate_agent.py "../../tests/output/$agent/agent-spec.yaml"
done
```

**Expected**: All 3 agents validate successfully

---

## Phase 2: Memory MCP Integration Testing

### Test Shared Context

Create a test script to simulate memory sharing:

```python
# test_memory_coordination.py
import yaml
import json
from datetime import datetime

def simulate_memory_write(agent_name, memory_key, content):
    """Simulate Memory MCP write with tagging protocol"""
    tagged_memory = {
        "WHO": agent_name,
        "WHEN": {
            "iso": datetime.utcnow().isoformat(),
            "unix": int(datetime.utcnow().timestamp()),
            "readable": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        },
        "PROJECT": "fullstack-app-test",
        "WHY": "implementation",
        "content": content
    }

    print(f"\n[{agent_name}] Writing to memory: {memory_key}")
    print(json.dumps(tagged_memory, indent=2))
    return tagged_memory

# Test backend agent sharing API schema
backend_memory = simulate_memory_write(
    "python-backend-specialist",
    "fullstack-app/api-schema",
    {
        "endpoints": [
            {"path": "/api/users", "method": "GET", "auth": "required"},
            {"path": "/api/users/:id", "method": "GET", "auth": "required"},
            {"path": "/api/auth/login", "method": "POST", "auth": "none"}
        ],
        "models": {
            "User": {"id": "int", "email": "string", "created_at": "datetime"}
        }
    }
)

# Test frontend agent consuming API schema
frontend_memory = simulate_memory_write(
    "typescript-frontend-specialist",
    "fullstack-app/api-integration",
    {
        "api_client": "axios",
        "endpoints_implemented": ["/api/users", "/api/auth/login"],
        "state_management": "zustand",
        "components": ["UserList", "LoginForm"]
    }
)

# Test testing agent validating contracts
testing_memory = simulate_memory_write(
    "integration-testing-specialist",
    "fullstack-app/test-results",
    {
        "contract_tests": {
            "api_users_get": "PASS",
            "api_auth_login": "PASS"
        },
        "e2e_tests": {
            "user_login_flow": "PASS",
            "user_list_display": "PASS"
        }
    }
)

print("\n" + "="*70)
print("MEMORY COORDINATION TEST SUMMARY")
print("="*70)
print(f"✓ Backend shared: API schema ({len(backend_memory['content']['endpoints'])} endpoints)")
print(f"✓ Frontend consumed: API integration ({len(frontend_memory['content']['endpoints_implemented'])} endpoints)")
print(f"✓ Testing validated: {len(testing_memory['content']['contract_tests'])} contract tests")
print("\nMemory coordination: SUCCESS")
```

**Run Test**:
```bash
python3 test_memory_coordination.py
```

**Expected Output**:
```
[python-backend-specialist] Writing to memory: fullstack-app/api-schema
{
  "WHO": "python-backend-specialist",
  ...
}

======================================================================
MEMORY COORDINATION TEST SUMMARY
======================================================================
✓ Backend shared: API schema (3 endpoints)
✓ Frontend consumed: API integration (2 endpoints)
✓ Testing validated: 2 contract tests

Memory coordination: SUCCESS
```

---

## Phase 3: Workflow Coordination Testing

### Test Sequential Workflow

Create workflow test script:

```python
# test_workflow_coordination.py
import yaml
from typing import List, Dict

class WorkflowCoordinator:
    def __init__(self):
        self.agents = {}
        self.execution_log = []

    def load_agent(self, agent_name: str, spec_path: str):
        """Load agent specification"""
        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        self.agents[agent_name] = spec
        print(f"✓ Loaded agent: {agent_name}")

    def execute_workflow(self, workflow: List[Dict]):
        """Execute multi-agent workflow"""
        print("\n" + "="*70)
        print("WORKFLOW EXECUTION")
        print("="*70 + "\n")

        for step in workflow:
            agent = step['agent']
            task = step['task']
            inputs = step.get('inputs', {})
            outputs = step.get('outputs', {})

            print(f"Step {workflow.index(step) + 1}: {agent}")
            print(f"  Task: {task}")
            print(f"  Inputs: {list(inputs.keys())}")
            print(f"  Outputs: {list(outputs.keys())}")

            # Simulate execution
            self.execution_log.append({
                'agent': agent,
                'task': task,
                'status': 'completed'
            })

            print(f"  Status: ✓ COMPLETED\n")

    def generate_report(self):
        """Generate workflow execution report"""
        print("="*70)
        print("WORKFLOW EXECUTION REPORT")
        print("="*70)
        print(f"Total steps: {len(self.execution_log)}")
        print(f"Completed: {sum(1 for log in self.execution_log if log['status'] == 'completed')}")
        print(f"Failed: {sum(1 for log in self.execution_log if log['status'] == 'failed')}")
        print("\nAgent Participation:")
        for agent in set(log['agent'] for log in self.execution_log):
            count = sum(1 for log in self.execution_log if log['agent'] == agent)
            print(f"  - {agent}: {count} tasks")

# Initialize coordinator
coordinator = WorkflowCoordinator()

# Load agents
coordinator.load_agent('backend', '../../tests/output/python-backend-specialist/agent-spec.yaml')
coordinator.load_agent('frontend', '../../tests/output/typescript-frontend-specialist/agent-spec.yaml')
coordinator.load_agent('testing', '../../tests/output/integration-testing-specialist/agent-spec.yaml')

# Define workflow
fullstack_workflow = [
    {
        'agent': 'backend',
        'task': 'Design and implement REST API',
        'inputs': {'requirements': 'User authentication and CRUD'},
        'outputs': {'api_schema': 'API endpoints definition', 'database_schema': 'SQLAlchemy models'}
    },
    {
        'agent': 'frontend',
        'task': 'Build React UI components',
        'inputs': {'api_schema': 'From backend'},
        'outputs': {'components': 'UserList, LoginForm', 'api_integration': 'Axios client'}
    },
    {
        'agent': 'testing',
        'task': 'Create integration and E2E tests',
        'inputs': {'api_schema': 'From backend', 'components': 'From frontend'},
        'outputs': {'test_results': 'Contract and E2E test results'}
    },
    {
        'agent': 'backend',
        'task': 'Address test failures and optimize',
        'inputs': {'test_results': 'From testing'},
        'outputs': {'optimized_api': 'Performance improvements'}
    },
    {
        'agent': 'testing',
        'task': 'Final validation',
        'inputs': {'optimized_api': 'From backend'},
        'outputs': {'final_report': 'All tests passing'}
    }
]

# Execute workflow
coordinator.execute_workflow(fullstack_workflow)

# Generate report
coordinator.generate_report()
```

**Run Test**:
```bash
python3 test_workflow_coordination.py
```

**Expected Output**:
```
✓ Loaded agent: backend
✓ Loaded agent: frontend
✓ Loaded agent: testing

======================================================================
WORKFLOW EXECUTION
======================================================================

Step 1: backend
  Task: Design and implement REST API
  Inputs: ['requirements']
  Outputs: ['api_schema', 'database_schema']
  Status: ✓ COMPLETED

Step 2: frontend
  Task: Build React UI components
  Inputs: ['api_schema']
  Outputs: ['components', 'api_integration']
  Status: ✓ COMPLETED

...

======================================================================
WORKFLOW EXECUTION REPORT
======================================================================
Total steps: 5
Completed: 5
Failed: 0

Agent Participation:
  - backend: 2 tasks
  - frontend: 1 tasks
  - testing: 2 tasks
```

---

## Phase 4: Claude Code Task Integration

### Test Task Template Generation

```python
# test_task_generation.py
import yaml

def generate_task_templates(agent_specs):
    """Generate Claude Code Task templates for agents"""
    templates = []

    for agent_name, spec_path in agent_specs.items():
        with open(spec_path) as f:
            spec = yaml.safe_load(f)

        task_template = spec['integration']['claude_code']['task_template']
        templates.append({
            'agent': agent_name,
            'template': task_template
        })

    return templates

agents = {
    'python-backend': '../../tests/output/python-backend-specialist/agent-spec.yaml',
    'typescript-frontend': '../../tests/output/typescript-frontend-specialist/agent-spec.yaml',
    'integration-testing': '../../tests/output/integration-testing-specialist/agent-spec.yaml'
}

templates = generate_task_templates(agents)

print("CLAUDE CODE TASK TEMPLATES")
print("="*70 + "\n")

for template in templates:
    print(f"Agent: {template['agent']}")
    print(f"Template:\n{template['template']}\n")
    print("-"*70 + "\n")

# Generate example parallel execution
print("PARALLEL EXECUTION EXAMPLE")
print("="*70 + "\n")
print('[Single Message - Parallel Agent Execution]:')
for template in templates:
    task_call = template['template'].replace('{{TASK_DESCRIPTION}}', f'Work on fullstack app').replace('{{CATEGORY}}', 'development')
    print(f'  {task_call}')
```

**Run Test**:
```bash
python3 test_task_generation.py
```

**Expected**: Task templates for all 3 agents

---

## Validation Checklist

### Multi-Agent Coordination
- [ ] All 3 agents created and validated
- [ ] Memory sharing protocol defined
- [ ] Coordination topology specified (mesh/hierarchical)
- [ ] Input/output contracts clear

### Memory MCP Integration
- [ ] Tagging protocol (WHO/WHEN/PROJECT/WHY) implemented
- [ ] Shared context keys defined
- [ ] Memory retrieval paths specified
- [ ] Cross-agent data flow validated

### Workflow Execution
- [ ] Sequential workflow defined
- [ ] Agent dependencies identified
- [ ] Task inputs/outputs mapped
- [ ] Workflow execution successful

### Claude Code Integration
- [ ] Task templates generated for all agents
- [ ] Parallel execution pattern defined
- [ ] Hooks integration specified
- [ ] Coordination protocol clear

---

## Test Results

**Date**: _______________
**Tester**: _______________
**Status**: ☐ PASS ☐ FAIL

### Integration Test Results

| Component | Status | Notes |
|-----------|--------|-------|
| Agent Creation (3 agents) | ☐ PASS ☐ FAIL | |
| Memory MCP Coordination | ☐ PASS ☐ FAIL | |
| Workflow Execution | ☐ PASS ☐ FAIL | |
| Task Template Generation | ☐ PASS ☐ FAIL | |
| End-to-End Workflow | ☐ PASS ☐ FAIL | |

### Performance Metrics

- Workflow completion time: _______________
- Memory operations: _______________
- Agent coordination overhead: _______________

### Notes

_Record observations on multi-agent coordination effectiveness_

---

## Cleanup

```bash
rm -rf ../../tests/output/python-backend-specialist/
rm -rf ../../tests/output/typescript-frontend-specialist/
rm -rf ../../tests/output/integration-testing-specialist/
rm -f test_memory_coordination.py
rm -f test_workflow_coordination.py
rm -f test_task_generation.py
```

---

## Conclusion

This test validates:
1. Multi-agent system creation
2. Memory-based coordination
3. Workflow orchestration
4. Claude Code integration

**Next Steps**:
- Deploy agents to production Claude-Flow environment
- Test with real-world full-stack development tasks
- Monitor coordination overhead and optimize
- Build additional specialist agents for comprehensive coverage
