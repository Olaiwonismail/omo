from tools.model import model
from tools.dynamic_prompt import prompt_with_context,get_lessons,get_quiz

from langchain.agents import create_agent

agent = create_agent(model, tools=[], middleware=[prompt_with_context])
tutor_agent = create_agent(model, tools=[], middleware=[get_lessons])
quiz_agent = create_agent(model ,tools=[],middleware=[get_quiz])
# query ="""what is Poiseuilles law """
# def ask_chatbot(query : str):

#     for step in agent.stream(
#         {"messages": [{"role": "user", "content": query}]},
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()

# def tutor(query : str):

#     for step in tutor_agent.stream(
#         {"messages": [{"role": "user", "content": query}]},
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()
async def tutor(query: str,adapt):
    query= f'Act as a clear, methodical {adapt}AI Tutor {query}'
    result = tutor_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    msgs = result["messages"]
    return msgs[-1].content

async def quiz():
    result = quiz_agent.invoke(
        {"messages": [{"role": "user", "content": " ho"}]}
    )
    msgs = result["messages"]
    return msgs[-1].content

