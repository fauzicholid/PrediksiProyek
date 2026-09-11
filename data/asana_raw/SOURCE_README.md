# Asana Enterprise Workspace Simulation Dataset

This dataset simulates a realistic Asana workspace for a large B2B SaaS company
(5,000–10,000 employees), designed for reinforcement learning and agent-based
interaction research.

## Contents

- `finaldb.sqlite` – Full relational SQLite database (recommended)
- `csv/` – Table-wise CSV exports for easy exploration:
  - organizations
  - teams
  - users
  - projects
  - sections
  - tasks
  - tags
  - task_tags
  - comments

## Key Features

- Realistic task naming & descriptions (LLM-assisted)
- Temporal consistency (creation, due dates, completion)
- Many-to-many relationships (teams, tags)
- Enterprise workflows (engineering, marketing, ops)

## Use Cases

- Reinforcement Learning environments
- Agent navigation & task automation
- Enterprise workflow simulation
- Graph & relational learning
