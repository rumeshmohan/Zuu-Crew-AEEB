"""
🏦 OPERATION LEDGER-MIND - SUBMISSION FILES CHECKER

Finalized checker: Removed optional progress files and instructor brief.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class FileType(Enum):
    """File type categories for better reporting"""
    REQUIRED = "🔴 Required"
    IMPORTANT = "🟡 Important"


@dataclass
class FileCheck:
    """File check specification"""
    path: str
    description: str
    file_type: FileType
    min_size_kb: float = 0.0
    assignment_section: str = ""


# Define expected submission files based on project requirements
SUBMISSION_FILES = [
    # ========== ROOT DOCUMENTATION & ENV ==========
    FileCheck("../README.md", "Project documentation", FileType.REQUIRED, 0.5, "General"),
    FileCheck("../pyproject.toml", "UV dependency configuration", FileType.IMPORTANT, 0.2, "General"),
    FileCheck("../uv.lock", "Locked dependencies", FileType.IMPORTANT, 1.0, "General"),
    
    # ========== SOURCE DATA ==========
    FileCheck(
        "../data/pdfs/2024-Annual-Report.pdf", 
        "Source document - Uber Technologies 2024 Annual Report", 
        FileType.REQUIRED, 1000, "Part 1: Data Factory"
    ),
    
    # ========== PART 1: DATA FACTORY OUTPUTS ==========
    FileCheck("../artifacts/data/train.jsonl", "Training Q&A pairs (80% split)", FileType.REQUIRED, 50, "Part 1"),
    FileCheck("../artifacts/data/golden_test_set.jsonl", "Test Q&A pairs (20% split)", FileType.REQUIRED, 20, "Part 1"),
    
    # ========== PART 2: FINE-TUNING OUTPUTS ==========
    FileCheck(
        "../artifacts/outputs/llama-3-financial-intern.zip", 
        "LoRA adapters (trained model weights)", 
        FileType.REQUIRED, 1000, "Part 2"
    ),
    
    # ========== PART 4: EVALUATION OUTPUTS ==========
    FileCheck("../artifacts/data/intern_predictions.jsonl", "Fine-tuned model predictions", FileType.REQUIRED, 20, "Part 4"),
    FileCheck("../artifacts/outputs/final_showdown.csv", "Evaluation results (ROUGE-L, LLM-Judge)", FileType.REQUIRED, 5, "Part 4"),
    
    # ========== NOTEBOOKS ==========
    FileCheck("../notebooks/01_data_factory.ipynb", "Data generation pipeline", FileType.REQUIRED, 5),
    FileCheck("../notebooks/02_finetuning_intern.ipynb", "LoRA fine-tuning", FileType.REQUIRED, 5),
    FileCheck("../notebooks/03_rag_librarian.ipynb", "Hybrid RAG system", FileType.REQUIRED, 5),
    FileCheck("../notebooks/04_evaluation_arena.ipynb", "Evaluation & Analysis", FileType.REQUIRED, 5),
    
    # ========== SOURCE CODE ==========
    FileCheck("../src/config/config.yaml", "System configuration", FileType.REQUIRED, 0.5),
    FileCheck("../src/config/prompts.yaml", "Prompt templates", FileType.REQUIRED, 0.5),
    FileCheck("../src/services/data_manager.py", "Data loading utilities", FileType.REQUIRED, 0.5),
    FileCheck("../src/services/llm_services.py", "LLM API wrappers", FileType.REQUIRED, 1.0),
    
    # ========== DELIVERABLE 6: ENGINEERING REPORT ==========
    FileCheck(
        "../Engineering_Report.pdf", 
        "Technical report (1500 words): Mark-heavy deliverable", 
        FileType.REQUIRED, 50, "Deliverable 6 - 15% MARKS"
    ),
]


def get_file_size_kb(filepath: Path) -> float:
    try:
        return filepath.stat().st_size / 1024
    except:
        return 0.0


def check_file(file_check: FileCheck, base_path: Path) -> Tuple[bool, str, float]:
    filepath = base_path / file_check.path
    if not filepath.exists():
        return False, "❌ MISSING", 0.0
    
    size_kb = get_file_size_kb(filepath)
    if filepath.is_file():
        if size_kb < file_check.min_size_kb:
            return False, f"⚠️  SMALL ({size_kb:.1f}KB)", size_kb
        return True, f"✅ OK ({size_kb:.1f}KB)", size_kb
    return False, "❌ NOT A FILE", 0.0


def print_section(title: str):
    print(f"\n{'─' * 80}\n  {title}\n{'─' * 80}")


def check_submission(base_path: Path = None) -> bool:
    base_path = base_path or Path.cwd()
    print("=" * 80)
    print("🏦 OPERATION LEDGER-MIND - SUBMISSION FILES CHECKER")
    print("=" * 80)
    print(f"📁 Root: {base_path.absolute()}\n")

    results = {ft: [] for ft in FileType}
    for fc in SUBMISSION_FILES:
        exists, status, size = check_file(fc, base_path)
        results[fc.file_type].append((fc, exists, status, size))

    for ft in FileType:
        print_section(f"{ft.value} FILES")
        for fc, exists, status, _ in results[ft]:
            print(f"  {status.ljust(25)} {fc.path.ljust(45)}")
            if not exists:
                print(f"     └─ {fc.description}")

    # Summary
    print_section("📊 SUMMARY")
    total = sum(len(v) for v in results.values())
    found = sum(sum(1 for _, ex, _, _ in v if ex) for v in results.values())
    size_mb = sum(sz for v in results.values() for _, ex, _, sz in v if ex) / 1024
    
    print(f"  Files Found: {found}/{total}")
    print(f"  Total Size:  {size_mb:.2f} MB")

    # Final Status
    req_missing = [fc for fc, ex, _, _ in results[FileType.REQUIRED] if not ex]
    
    print_section("🎯 SUBMISSION STATUS")
    if not req_missing:
        print("\n  ✅ READY TO SUBMIT!")
        print("  Ensure you have zipped the entire project root.")
        return True
    else:
        print(f"\n  ❌ NOT READY: Missing {len(req_missing)} required files.")
        for m in req_missing:
            print(f"     - {m.path}")
        return False


if __name__ == "__main__":
    success = check_submission()
    sys.exit(0 if success else 1)