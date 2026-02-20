from langchain_core.prompts import ChatPromptTemplate

email_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert executive assistant who extracts "
            "structured action items from emails.\n"
            "You MUST follow the provided JSON schema exactly.\n"
            "Return only structured data.\n"
            "Be precise and conservative.\n"
            "Do not hallucinate missing information."
        ),
        (
            "human",
            """
Extract all actionable tasks from the email.

Return STRICT JSON in this format:

{{
  "actions": [
    {{
      "task": "",
      "owner": "",
      "deadline": "",
      "priority": "",
      "source_sentence": ""
    }}
  ],
  "overall_urgency": "",
  "email_type": ""
}}

Rules:
- If owner missing → null
- If deadline missing → null
- Priority: High, Medium, Low

EMAIL:
{email_text}
"""
        ),
    ]
)