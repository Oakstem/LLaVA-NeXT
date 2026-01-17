"""GPT-based structured extraction of gaze information from text descriptions.

This module provides functionality to extract structured gaze information
from LLaVA model responses using OpenAI's GPT API.
"""

import json
from openai import OpenAI


# GPT extraction schema for structured outputs
GPT_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "gaze_interaction_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "persons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "person_id": {"type": "integer"},
                            "description": {"type": "string"},
                            "attention_focus": {"type": ["string", "null"]},
                        },
                        "required": ["person_id", "description", "attention_focus"],
                        "additionalProperties": False
                    }
                },
                "social_interaction_label": {
                    "type": ["string", "null"],
                    "enum": ["Mutual", "Single", "Joint attention", "Non-communicative", null]
                }
            },
            "required": ["persons", "social_interaction_label"],
            "additionalProperties": False
        },
        "strict": True
    }
}

GPT_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured gaze information from text descriptions.
Given a description of people in an image and their gaze behavior, extract:

1. For each person mentioned:
   - description: Their appearance/role description
   - attention_focus: What they are explicitly described as looking at. Set to null if no explicit looking direction/target is mentioned for this person.
   - person_id: Only include a numeric ID if the text explicitly identifies WHO this person is (e.g., "Person 1", "the first person") or provides clear distinguishing characteristics that could serve as an identifier. Do not include the person is not explicitly identified or distinguished.

2. social_interaction_label: If an EXPLICIT social interaction label is provided in the text (e.g., "Social interaction label: Mutual gaze (A→B and B→A)"), normalize it to one of: "Mutual", "Single", "Joint attention", "Non-communicative". Set to null if no explicit label is provided in the text.

Reply with JSON only, following the schema exactly."""


def extract_gaze_info_with_gpt(
    client: OpenAI,
    response_text: str,
    gpt_model: str,
) -> dict | None:
    """Extract structured gaze information from response text using GPT.
    
    Args:
        client: OpenAI client instance
        response_text: The LLaVA model's response text to extract from
        gpt_model: OpenAI model to use for extraction
        
    Returns:
        Extracted gaze information dict or None if extraction fails
    """
    if not response_text or not response_text.strip():
        return None
    
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": GPT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": response_text},
        ],
        response_format=GPT_EXTRACTION_SCHEMA,
    )
    
    content = completion.choices[0].message.content
    return json.loads(content)
