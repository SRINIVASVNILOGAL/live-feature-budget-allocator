# Live Feature Budget Allocator:
An RL environment where an AI agent learns to allocate
a fixed daily budget across AI product features to maximize
user retention while minimizing cost overruns.
---
## What This Environment Does
An AI product has 3 features:
- Chat
- Autocomplete
- AI Agent

Each feature costs money every time a user uses it.
Each feature affects whether users stay or leave.

The AI agent learns to decide:
- Which feature gets more budget
- Which feature gets cut when money is tight
- How to react when conditions change dynamically
---
## The RL Loop
reset()  → start fresh episode
step()   → agent picks action, environment reacts
state()  → returns current episode metadata
---
## Action Space
| Action | Description |
|--------|-------------|
| 0 | Fund Chat more |
| 1 | Fund Autocomplete more |
| 2 | Fund AI Agent more |
| 3 | Cut worst performing feature |
---
## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| features | list | All features with cost, success, retention, usage |
| budget_remaining | float | How much budget is left |
| step | int | Current step number |
| done | bool | Is episode over |
| reward | float | Score from last action 0.0 to 1.0 |
---
## Reward Function
reward = 0.6 * retention
+ 0.3 * budget_efficiency
+ 0.1 * success_rate
Always clipped between 0.0 and 1.0
---

## Tasks
### Easy
- 2 features: Chat + Autocomplete
- Stable environment
- 5 steps
- Budget: $1000
### Medium
- 3 features: Chat + Autocomplete + Agent
- Slight drift each step
- 8 steps
- Budget: $800
### Hard
- 3 features: Chat + Autocomplete + Agent
- Dynamic changes every step
- Random spike events
- Budget shrinks $50 every step
- 10 steps
---

## Setup and Installation
```bash
pip install -r requirements.txt
```
---
## Run Locally
### Start the server
```bash
uvicorn env.app:app --host 0.0.0.0 --port 7860
```

### Run inference
```bash
python inference.py
```
## Run with Docker
### Build
```bash
docker build -t live-feature-budget-allocator .
```
### Run
```bash
docker run -p 7860:7860 live-feature-budget-allocator
```


## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| API_BASE_URL | Yes | LLM API endpoint |
| MODEL_NAME | Yes | Model to use for inference |
| HF_TOKEN | Yes | Hugging Face token |
| SERVER_URL | No | Environment server URL |
| TASK | No | Task difficulty easy/medium/hard |


## Expected Output
==================================================
FINAL RESULTS
Easy   score : 0.7232
Medium score : 0.6994
Hard   score : 0.6712
FINAL  score : 0.6979

## Project Structure
live_feature_budget_allocator/
├── env/
│   ├── models.py        ← Data types
│   ├── environment.py   ← Game logic
│   ├── app.py           ← Web server
│   └── client.py        ← HTTP client
├── inference.py         ← Test driver
├── requirements.txt     ← Libraries
├── Dockerfile           ← Container
├── openenv.yaml         ← Config
└── README.md            ← This file

