import json
from src.lastminute_api.application.agent_service.node import custom_handoff_tool

# Fix: Pass None or omit the parameter entirely 
test = custom_handoff_tool.invoke({
    "agent_name": "tavily_agent",
    "task_input": "test",
    "state_update_json": None  # This fixes the validation error
})

print(f"Result: {test}")
print(f"Command goto: {test.goto}")
print(f"Command update: {test.update}")
