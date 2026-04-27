"""
PersonaMem Converter - convert PersonaMem dataset to Locomo format.
"""
import json
import csv
import re
import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


def extract_persona_name(system_content: str) -> str:
    """Extract persona name from system message."""
    match = re.search(r'Name:\s*([^\n]+)', system_content)
    if match:
        return match.group(1).strip()
    return "User"


def clean_message_prefix(text: str) -> str:
    """Clean 'User:' and 'Assistant:' prefixes from messages."""
    text = re.sub(r'^(User|Assistant):\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_options(options_str: str) -> Dict[str, str]:
    """Parse all_options string, return dict."""
    try:
        options_list = ast.literal_eval(options_str)
        options_dict = {}
        for opt in options_list:
            match = re.match(r'\(([a-z])\)\s*(.*)', opt, re.DOTALL)
            if match:
                key = f"({match.group(1)})"
                value = match.group(2).strip()
                options_dict[key] = value
        return options_dict
    except Exception as e:
        print(f"Warning: Failed to parse options: {e}")
        return {}


@register_converter("personamem")
class PersonaMemConverter(BaseConverter):
    """PersonaMem dataset converter."""
    
    def get_input_files(self) -> Dict[str, str]:
        """Return required input files."""
        return {
            "questions": "questions_32k.csv",
            "contexts": "shared_contexts_32k.jsonl"
        }
    
    def get_output_filename(self) -> str:
        """Return output filename."""
        return "personamem_32k_locomo_style.json"
    
    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        """
        Execute conversion.
        
        Args:
            input_paths: {
                "questions": "path/to/questions_32k.csv",
                "contexts": "path/to/shared_contexts_32k.jsonl"
            }
            output_path: Output file path
        """
        print(f"🔄 Converting PersonaMem to Locomo format...")
        
        # 1. Read JSONL file, build context dict
        print("   Loading shared contexts...")
        contexts = {}
        with open(input_paths["contexts"], 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                contexts.update(data)
        print(f"   Loaded {len(contexts)} shared contexts")
        
        # 2. Read CSV file
        print("   Loading questions...")
        questions = []
        with open(input_paths["questions"], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            questions = list(reader)
        print(f"   Loaded {len(questions)} questions")
        
        # 3. Group by (shared_context_id, end_index_in_shared_context)
        print("   Grouping questions...")
        grouped_questions = defaultdict(list)
        for q in questions:
            key = (q['shared_context_id'], int(q['end_index_in_shared_context']))
            grouped_questions[key].append(q)
        print(f"   Grouped into {len(grouped_questions)} unique context groups")
        
        # 4. Convert to Locomo format
        print("   Converting to Locomo format...")
        locomo_data = []
        
        for (context_id, end_index), question_list in grouped_questions.items():
            # Get corresponding context
            if context_id not in contexts:
                print(f"   Warning: context_id {context_id} not found")
                continue
            
            full_context = contexts[context_id]
            context_messages = full_context[:end_index + 1]
            
            # Extract persona name
            persona_name = "User"
            assistant_name = "Assistant"
            if context_messages and context_messages[0]['role'] == 'system':
                persona_name = extract_persona_name(context_messages[0]['content'])
            
            # Create Locomo entry
            locomo_entry = {
                "qa": [],
                "conversation": {
                    "speaker_a": persona_name,
                    "speaker_b": assistant_name,
                    "session_0_date_time": "Unknown",  # PersonaMem lacks timestamp info
                    "session_0": []
                }
            }
            
            # Add all questions to qa list
            for q in question_list:
                options = parse_options(q['all_options'])
                correct_answer_text = options.get(q['correct_answer'], q['correct_answer'])
                
                qa_item = {
                    "question_id": q['question_id'],
                    "question": q['user_question_or_message'],
                    "answer": q['correct_answer'],
                    "answer_text": correct_answer_text,
                    "all_options": options,
                    "evidence": [],
                    "category": q['question_type'],  # Keep original type, no conversion
                    "topic": q['topic'],
                    "persona_id": q['persona_id'],
                    "context_length_in_tokens": int(q['context_length_in_tokens']),
                    "distance_to_ref_in_tokens": int(q['distance_to_ref_in_tokens']),
                }
                locomo_entry["qa"].append(qa_item)
            
            # Build dialogue list
            dialogue_idx = 0
            for msg in context_messages:
                if msg['role'] == 'system':
                    continue  # Skip system message
                
                speaker = persona_name if msg['role'] == 'user' else assistant_name
                cleaned_text = clean_message_prefix(msg['content'])
                
                dialogue_item = {
                    "speaker": speaker,
                    "text": cleaned_text,
                    "dia_id": f"D0:{dialogue_idx}"
                }
                locomo_entry["conversation"]["session_0"].append(dialogue_item)
                dialogue_idx += 1
            
            locomo_data.append(locomo_entry)
        
        # 5. Save result
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(locomo_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved {len(locomo_data)} entries to {output_path}")
        print(f"   Total questions: {sum(len(entry['qa']) for entry in locomo_data)}")

