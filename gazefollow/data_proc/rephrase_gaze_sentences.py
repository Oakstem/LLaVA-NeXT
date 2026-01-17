#!/usr/bin/env python3
"""
Rephrase gaze/attention descriptions to ensure grammatical correctness.

This script reads a JSON file containing gaze descriptions and uses Ollama
to rephrase sentences that incorrectly attribute gaze to objects/clothing
(e.g., "the glasses is looking at...") to proper person-centric descriptions
(e.g., "the person wearing glasses is looking at...").
"""

import json
import logging
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GazeRephraser:
    """Rephrase gaze descriptions using Ollama API."""
    
    def __init__(self, 
                 model: str = "gpt-oss:latest",
                 host: str = "127.0.0.1",
                 port: int = 11435,
                 timeout: int = 60):
        """
        Initialize the gaze rephraser.
        
        Args:
            model: The Ollama model to use
            host: Ollama server host
            port: Ollama server port
            timeout: Request timeout in seconds
        """
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for gaze sentence rephrasing."""
        prompt = """You are a grammar correction specialist for gaze/attention descriptions.
            Your task is to rephrase sentences to ensure they correctly describe ONE PERSON looking at something/someone.

            Common errors to fix:
            1. Objects/clothing as subjects: "the glasses is looking at..." → "the person wearing glasses is looking at..."
            2. Wearables as subjects: "the hat is looking at..." → "the person wearing the hat is looking at..."
            3. Subject-verb agreement: "the man are looking at..." → "the man is looking at..."
            4. Multiple people in description: Remove any additional people mentioned - describe ONLY the target person
            5. Incomplete descriptions: ensure the sentence is complete and logical.

            CRITICAL Rules:
            - Describe ONLY ONE PERSON (the target person from the question)
            - Always make that PERSON the subject who is looking
            - Keep the original gaze target for that person ONLY when it is present and unambiguous
            - Use proper subject-verb agreement
            - Keep descriptions concise and natural
            - If the sentence mentions multiple people, keep ONLY the target person's description
            - Preserve the specific details about the target person's appearance
            - Do NOT invent details about the gaze target or the person beyond what is present in the input
            - NEVER end the sentence with "is looking" without a gaze target.
            - If the subject is undefined, nonsensical, or clearly not a person (e.g., "a black item", "the a black item", "something", "an object") and it cannot be confidently resolved into "the person wearing/holding X", then do NOT attempt to invent a person — return the exact string "None".

            Output ONLY the rephrased sentence about the single target person, or the exact string "None" when the input cannot be resolved to a person. No explanations or additional text."""
        return prompt
    
    def _extract_target_person(self, human_question: str) -> str:
        """
        Extract the target person description from the human question.
        
        Args:
            human_question: The human question (e.g., "Describe where the man in white shirt is looking at")
            
        Returns:
            The person description (e.g., "the man in white shirt")
        """
        # Remove <image> token and clean up
        question = human_question.replace("<image>", "").strip()
        
        # Common patterns: "Describe where X is looking at" or "Where is X looking"
        import re
        
        # Try pattern: "Describe where X is looking"
        match = re.search(r'(?:Describe\s+)?where\s+(.+?)\s+is\s+looking', question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try pattern: "where X looks"
        match = re.search(r'where\s+(.+?)\s+looks', question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try pattern: "Where is X looking"
        match = re.search(r'Where\s+is\s+(.+?)\s+looking', question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: return the question without the verb parts
        return question.replace("Describe where", "").replace("Where is", "").replace("is looking at", "").replace("looking at", "").strip()
    
    def _create_user_prompt(self, sentence: str, target_person: str = None) -> str:
        """
        Create user prompt for a gaze description sentence.
        
        Args:
            sentence: The original gaze description
            target_person: The target person from the human question (optional)
            
        Returns:
            Formatted user prompt string
        """
        if target_person:
            prompt = f"""Rephrase this gaze description if needed:

                        "{sentence}"

                        The target person is: "{target_person}"

                        Describe ONLY where {target_person} is looking. Remove any mentions of other people.
                        Return only the corrected sentence about {target_person}."""
        else:
            prompt = f"""Rephrase this gaze description if needed:

                        "{sentence}"

                        Return only the corrected sentence."""
        
        return prompt
    
    def rephrase(self, sentence: str, target_person: str = None) -> str:
        """
        Rephrase a gaze description using Ollama API.
        
        Args:
            sentence: The original gaze description
            target_person: The target person from the human question (optional)
            
        Returns:
            Rephrased sentence
        """
        url = f"{self.base_url}/api/chat"
        user_prompt = self._create_user_prompt(sentence, target_person)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.5,  # Lower temperature for consistent rephrasing
                "top_p": 0.9,
                "top_k": 40,
                "think": "low"
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            rephrased = result.get("message", {}).get("content", "").strip()
            
            # Remove quotes if the model wrapped the response
            if rephrased.startswith('"') and rephrased.endswith('"'):
                rephrased = rephrased[1:-1]
            if rephrased.startswith("'") and rephrased.endswith("'"):
                rephrased = rephrased[1:-1]
            
            return rephrased if rephrased else sentence
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return sentence  # Return original on error
    
    def process_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single entry from the dataset.
        
        Args:
            entry: Dictionary containing conversation data
            
        Returns:
            Entry with rephrased gaze description
        """
        # Create a copy of the entry
        result = entry.copy()
        
        # Extract target person from human question
        target_person = None
        if "conversations" in entry:
            for conv in entry["conversations"]:
                if conv.get("from") == "human":
                    human_question = conv.get("value", "")
                    if human_question:
                        target_person = self._extract_target_person(human_question)
                        logger.debug(f"ID {entry.get('id')}: Extracted target person: '{target_person}'")
                    break
        
        # Process conversations
        bad_result = False
        if "conversations" in entry:
            conversations = []
            for conv in entry["conversations"]:
                conv_copy = conv.copy()
                
                # Rephrase GPT responses (the gaze descriptions)
                if conv.get("from") == "gpt":
                    original_value = conv.get("value", "")
                    if original_value:
                        rephrased_value = self.rephrase(original_value, target_person)
                        conv_copy["value"] = rephrased_value
                        
                        # Check for bad results
                        if rephrased_value == "None":
                            bad_result = True

                        # Log if changed
                        if rephrased_value != original_value:
                            logger.debug(f"ID {entry.get('id')}: '{original_value}' → '{rephrased_value}'")
                
                conversations.append(conv_copy)
            
            result["conversations"] = conversations
        
        if bad_result:
            logger.warning(f"ID {entry.get('id')}: Rephrasing resulted in 'None'")
            result = None
        return result
    
    def process_file(self, input_path: str, output_path: str = None, checkpoint_every: int = 500) -> None:
        """
        Process an entire JSON file with periodic checkpointing.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file (auto-generated if None)
            checkpoint_every: Save checkpoint every N entries (default: 500)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = input_path.parent / f"{input_path.stem}_rephrased_{timestamp}.json"
        else:
            output_path = Path(output_path)
        
        # Checkpoint path
        checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
        
        logger.info(f"Loading data from {input_path}")
        
        # Load input data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Processing {len(data)} entries (checkpointing every {checkpoint_every} entries)")
        
        # Process each entry
        results = []
        errors = 0
        missed = 0
        
        for idx, entry in enumerate(tqdm(data, desc="Rephrasing gaze descriptions"), start=1):
            try:
                result = self.process_entry(entry)
                if result is not None:
                    results.append(result)
                else:
                    missed += 1
            except Exception as e:
                logger.error(f"Failed to process entry {entry.get('id')}: {e}")
                errors += 1
                # Keep original entry on error
                results.append(entry)
            
            # Save checkpoint periodically
            if idx % checkpoint_every == 0:
                logger.info(f"Saving checkpoint at {idx}/{len(data)} entries to {checkpoint_path}")
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save final results
        logger.info(f"Saving final results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Remove checkpoint file on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Removed checkpoint file")
        
        logger.info(f"Processing complete!")
        logger.info(f"Total entries: {len(data)}")
        logger.info(f"Successful: {len(results)}")
        logger.info(f"Missed (None results): {missed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Output saved to: {output_path}")


def main():
    """Main entry point.

    Behavior:
    - Use OLLAMA_HOST environment variable if set (e.g. http://<ip>:11435 or just an IP)
    - Fall back to 127.0.0.1, then to WSL gateway, then to common WSL->Windows host IP 10.255.255.254
    - Accept CLI args for input/output/host/port/timeout
    """

    parser = argparse.ArgumentParser(description="Rephrase gaze descriptions for grammatical correctness")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("-o", "--output", help="Path to output JSON file (optional)")
    parser.add_argument("--host", help="Ollama host (IP or hostname). Overrides OLLAMA_HOST env var")
    parser.add_argument("--port", type=int, default=11435, help="Ollama port (default: 11435)")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N entries (default: 100)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    # Determine host
    env_host = os.environ.get("OLLAMA_HOST")
    host_candidate: Optional[str] = None
    if args.host:
        host_candidate = args.host
    elif env_host:
        # allow env var in forms like "http://127.0.0.1:11435" or "127.0.0.1"
        if env_host.startswith("http://") or env_host.startswith("https://"):
            host_candidate = env_host.replace("http://", "").replace("https://", "")
            # strip optional port
            host_candidate = host_candidate.split(":")[0]
        else:
            host_candidate = env_host
    else:
        host_candidate = None

    tried_hosts = []
    final_host = None

    candidates = []
    if host_candidate:
        candidates.append(host_candidate)
    
    # Add WSL default gateway (Windows host) detection
    try:
        import subprocess
        result = subprocess.run(
            ["ip", "route", "show"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split("\n"):
            if "default" in line:
                parts = line.split()
                if len(parts) >= 3:
                    gateway = parts[2]
                    candidates.append(gateway)
                    logger.debug(f"Detected WSL gateway: {gateway}")
                break
    except Exception as e:
        logger.debug(f"Could not detect WSL gateway: {e}")
    
    candidates.extend(["127.0.0.1", "10.255.255.254"])

    for h in candidates:
        url = f"http://{h}:{args.port}/api/tags"
        tried_hosts.append(url)
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                final_host = h
                logger.info(f"✓ Found Ollama at {final_host}:{args.port}")
                break
        except requests.RequestException:
            continue

    if final_host is None:
        # Last resort: use provided host_candidate or 127.0.0.1
        final_host = host_candidate or "127.0.0.1"
        logger.warning("Could not auto-detect reachable Ollama host. Tried: %s. Falling back to %s:%s",
                       tried_hosts, final_host, args.port)

    input_file = args.input

    # Initialize rephraser
    rephraser = GazeRephraser(
        model="gpt-oss:latest",
        host=final_host,
        port=args.port,
        timeout=args.timeout
    )

    # Process the file
    try:
        rephraser.process_file(input_file, output_path=args.output, checkpoint_every=args.checkpoint_every)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
