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
                            "gaze_target": {"type": ["string", "null"]},
                        },
                        "required": ["person_id", "description", "gaze_target"],
                        "additionalProperties": False
                    }
                },
                "inferred_gaze_interaction": {
                    "type": "string",
                    "enum": ["non-communicative gaze", "mutual gaze", "joint attention toward a shared object"]
                }
            },
            "required": ["persons", "inferred_gaze_interaction"],
            "additionalProperties": False
        },
        "strict": True
    }
}

GPT_EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured gaze information from text descriptions.
Given a description of people in an image and their gaze behavior, extract:
1. For each person mentioned: their description and what they are looking at (gaze target)
2. The overall gaze interaction type: "non-communicative gaze", "mutual gaze", or "joint attention toward a shared object"

Reply with JSON only, following the schema exactly. Number persons in the order they appear in the description."""


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
