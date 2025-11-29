# Tutorial: codes_2

The **Agent Development Kit (ADK)** is a framework for building, orchestrating, and deploying **AI agents** - autonomous systems that can reason, take actions, and collaborate to solve complex problems. Agents use **Tools** to interact with external systems (like APIs and databases), **Multi-Agent Systems** enable specialized agents to work together through patterns like sequential, parallel, and loop workflows, and **Memory Management** (via Sessions for short-term context and Memory Services for long-term knowledge) allows agents to maintain context across conversations. The framework includes **A2A Communication** for cross-organization agent integration and provides **Runners** and **Apps** to orchestrate execution, with support for **Callbacks** to customize agent lifecycle behavior.


**Source Repository:** [None](None)

```mermaid
flowchart TD
    A0["Agent
"]
    A1["Tools
"]
    A2["Multi-Agent Systems
"]
    A3["Session Management
"]
    A4["Memory Service
"]
    A5["Runner
"]
    A6["Agent2Agent (A2A) Communication
"]
    A7["Workflow Patterns
"]
    A8["Context Engineering
"]
    A9["Long-Running Operations
"]
    A10["Callbacks
"]
    A0 -- "Invokes" --> A1
    A0 -- "Composes into" --> A2
    A5 -- "Orchestrates" --> A0
    A5 -- "Manages" --> A3
    A5 -- "Integrates with" --> A4
    A3 -- "Transfers to" --> A4
    A2 -- "Implements" --> A7
    A0 -- "Communicates via" --> A6
    A0 -- "Applies" --> A8
    A1 -- "Can be" --> A9
    A5 -- "Triggers" --> A10
```

## Chapters

1. [Agent
](01_agent_.md)
2. [Tools
](02_tools_.md)
3. [Multi-Agent Systems
](03_multi_agent_systems_.md)
4. [Workflow Patterns
](04_workflow_patterns_.md)
5. [Agent2Agent (A2A) Communication
](05_agent2agent__a2a__communication_.md)
6. [Runner
](06_runner_.md)
7. [Session Management
](07_session_management_.md)
8. [Memory Service
](08_memory_service_.md)
9. [Context Engineering
](09_context_engineering_.md)
10. [Callbacks
](10_callbacks_.md)
11. [Long-Running Operations
](11_long_running_operations_.md)

