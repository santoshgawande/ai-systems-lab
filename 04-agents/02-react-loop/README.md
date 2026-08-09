# Lab 02 — ReAct Loop

Reason + Act: the core pattern behind Claude Code, Devin, and every agentic system.

## What you learn

- How the Thought/Action/Observation loop works step by step
- How to parse the model's structured output and route to tool functions
- How stop conditions (max steps, "Final Answer") prevent infinite loops
- Why the observation is fed back as the next user message

## Run

```bash
python react.py "calculate the area of a circle with radius 7 then write the result to /tmp/result.txt"
python react.py "read /etc/hostname and tell me what it says"
python react.py "list what is in the current directory"
```

## The ReAct format

The model must respond in this exact structure every turn:
```
Thought: I need to calculate pi * r^2 where r = 7
Action: calculator
Action Input: {"expression": "3.14159 * 7 * 7"}
```

Then you execute the tool, and send back:
```
Observation: 153.93804...
```

The model reads the observation and continues until it outputs `Final Answer:`.
