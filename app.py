from tools.model import model
from tools.dynamic_prompt import prompt_with_context

from langchain.agents import create_agent

agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query ="""what is Poiseuilles law """
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()