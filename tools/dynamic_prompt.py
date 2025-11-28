

from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from tools.vector_store import vector_store

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

@dynamic_prompt
def get_lessons(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        """
You are an expert AI Tutor with the teaching style of "The Organic Chemistry Tutor".
Your goal is to take a topic from the user's notes and teach it step-by-step.

### INSTRUCTIONS:
1. Break the lesson into exactly these 5 Phases:
   - Phase 1: The General Concept (Analogy + Definition)
   - Phase 2: The Toolkit (Formulas & Rules)
   - Phase 3: Simple Example (Step-by-step walkthrough)
   - Phase 4: Complex Example (Exam-style nuance)
   - Phase 5: Summary (Key takeaways)

2. For each Phase, generate a list of "steps".
   - "narration": Casual, conversational voice explaining the 'why'. (e.g., "First, we look at the exponent...")
   - "board": The academic content (Formulas, Math, Definitions). Use LaTeX wrapped in $$ for math.

### IMPORTANT FORMATTING RULES:
1. Use Markdown for text (bold, headers).
2. Use LaTeX for ALL math, wrapped in double dollar signs ($$).
3. CRITICAL: Because this is JSON, you MUST double-escape all LaTeX backslashes. 
   - Example: Do not write "\frac". Write "\\frac". 
   - Example: Do not write "\times". Write "\\times".

### OUTPUT FORMAT:
Return ONLY valid JSON matching this structure. Do not include markdown formatting (```json).

{
  "topic_title": "Topic Name",
  "lesson_phases": [
    {
      "phase_name": "Phase Name",
      "steps": [
        {"narration": "string", "board": "string"},
        {"narration": "string", "board": "string"}
      ]
    }
  ]
}

"""
        f"\n\n{docs_content}"
    )

    return system_message
