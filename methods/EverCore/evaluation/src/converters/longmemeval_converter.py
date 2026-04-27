"""
LongMemEval Converter - convert LongMemEval dataset to Locomo format.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


def convert_time_format(input_str: str) -> str:
    """
    Convert time string from "YYYY/MM/DD (Day) HH:MM" format
    to "H:MM am/pm on D Month, YYYY" format.
    """
    # Input format: %Y: year, %m: month, %d: day, %a: weekday abbr, %H: 24-hour, %M: minute
    input_format = "%Y/%m/%d (%a) %H:%M"
    
    # Parse input string to datetime object
    dt_object = datetime.strptime(input_str, input_format)
    
    # Output format: %-I: 12-hour (no leading zero), %M: minute, %p: AM/PM, 
    #                %-d: day (no leading zero), %B: full month name, %Y: year
    output_format = "%-I:%M %p on %-d %B, %Y"
    
    # Format to target string and convert AM/PM to lowercase
    formatted_string = dt_object.strftime(output_format).lower()
    
    # Ensure month is capitalized
    parts = formatted_string.split(' ')
    parts[4] = parts[4].capitalize()
    
    return ' '.join(parts)


def convert_lmeval_s_to_locomo_style(lmeval_data: list) -> list:
    """
    Convert LongMemEval-S format to Locomo format.
    
    Args:
        lmeval_data: LongMemEval-S raw data
        
    Returns:
        Locomo format data
    """
    locomo_style_data = []
    
    for data in lmeval_data:
        data_dict = {
            "qa": [],
            "conversation": {}
        }
        
        # Find session indices containing answers
        evidence_session_idx = []
        for idx, session_id in enumerate(data["haystack_session_ids"]):
            if session_id in data["answer_session_ids"]:
                evidence_session_idx.append(idx)
        
        # Mark messages containing answers
        for idx, session in enumerate(data["haystack_sessions"]):
            for i, msg in enumerate(session):
                data["haystack_sessions"][idx][i]["has_answer"] = idx in evidence_session_idx
        
        # Collect evidence
        evidence = []
        for idx, session in enumerate(data["haystack_sessions"]):
            for i, msg in enumerate(session):
                if msg["has_answer"]:
                    evidence.append(f"D{idx}:{i}")
        
        # Build QA
        data_dict["qa"].append({
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"],
            "evidence": evidence,
            "category": data["question_type"]
        })
        
        # Build conversation
        data_dict["conversation"]["speaker_a"] = f"user_{data['question_id']}"
        data_dict["conversation"]["speaker_b"] = f"assistant_{data['question_id']}"
        
        for idx, session in enumerate(data["haystack_sessions"]):
            data_dict["conversation"][f"session_{idx}_date_time"] = convert_time_format(
                data["haystack_dates"][idx]
            )
            data_dict["conversation"][f"session_{idx}"] = []
            
            for i, msg in enumerate(session):
                data_dict["conversation"][f"session_{idx}"].append({
                    "speaker": msg["role"] + f"_{data['question_id']}",
                    "text": msg["content"],
                    "dia_id": f"D{idx}:{i}"
                })
        
        locomo_style_data.append(data_dict)
    
    return locomo_style_data


@register_converter("longmemeval")
class LongMemEvalConverter(BaseConverter):
    """LongMemEval dataset converter."""
    
    def get_input_files(self) -> Dict[str, str]:
        """Return required input files."""
        return {
            "raw": "longmemeval_s_cleaned.json"
        }
    
    def get_output_filename(self) -> str:
        """Return output filename."""
        return "longmemeval_s_locomo_style.json"
    
    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        """
        Execute conversion.
        
        Args:
            input_paths: {"raw": "path/to/longmemeval_s_cleaned.json"}
            output_path: Output file path
        """
        print(f"🔄 Converting LongMemEval to Locomo format...")
        
        # Read raw data
        with open(input_paths["raw"], "r", encoding="utf-8") as f:
            lmeval_data = json.load(f)
        
        print(f"   Loaded {len(lmeval_data)} items")
        
        # Convert format
        locomo_style_data = convert_lmeval_s_to_locomo_style(lmeval_data)
        
        # Save result
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_style_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved {len(locomo_style_data)} entries to {output_path}")

