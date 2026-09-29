import json
import sys
# import os

from agent_app import run_agent_pipeline
question = sys.argv[1]
answer = run_agent_pipeline(question, 'gemini')

print(json.dumps({"output": answer}))
