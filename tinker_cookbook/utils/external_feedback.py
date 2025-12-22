import os
from google import genai
from google.genai import types

async def get_external_feedback(prompt: str, feedback_temperature: float = 0.0, feedback_max_tokens: int = 2048, model: str = "gemini-2.0-flash-lite") -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=feedback_temperature,
            max_output_tokens=feedback_max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
    )
    return response.text

def extract_external_feedback(text: str, start_tag: str, end_tag: str) -> str:
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    if start_idx == -1:
        return text
    elif end_idx == -1:
        return text[start_idx + len(start_tag):]
    else:
        return text[start_idx + len(start_tag):end_idx]