#!/usr/bin/env python3
"""
Translation script using DeepL API for hybrid text classification project.

This script translates texts from one language to another using the DeepL API,
manages annotations, and integrates translated texts into the unified dataset.

Requirements:
    - DeepL API key set as DEEPL_AUTH_KEY environment variable
    - deepl-python library installed: pip install deepl
    
Usage:
    python translate_texts_deepl.py --source EN --target PT
    python translate_texts_deepl.py --source EN --target RU --batch-size 50
    python translate_texts_deepl.py --source EN --target BG --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import csv
import time
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

try:
    import deepl
except ImportError:
    print("Error: deepl-python library not found. Install with: pip install deepl")
    sys.exit(1)

# Language code mappings for DeepL API
DEEPL_LANG_CODES = {
    'EN': 'EN',
    'PT': 'PT-BR',  # Brazilian Portuguese
    'RU': 'RU',
    'BG': 'BG',
    'HI': 'HI',
}

LANG_NAMES = {
    'EN': 'English',
    'PT': 'Portuguese',
    'RU': 'Russian', 
    'BG': 'Bulgarian',
    'HI': 'Hindi',
}


@dataclass
class TextAnnotation:
    """Represents a text annotation with narratives and subnarratives."""
    file_id: str
    narratives: str
    subnarratives: str


@dataclass
class TranslationJob:
    """Represents a translation job for a single text."""
    file_id: str
    original_text: str
    annotation: TextAnnotation
    target_file_path: str


class DeepLTranslator:
    """Manages DeepL API translations and dataset integration."""
    
    def __init__(self, source_lang: str, target_lang: str, batch_size: int = 100, dry_run: bool = False):
        self.source_lang = source_lang.upper()
        self.target_lang = target_lang.upper()
        self.batch_size = batch_size
        self.dry_run = dry_run
        
        # Validate language codes
        if self.source_lang not in DEEPL_LANG_CODES:
            raise ValueError(f"Source language '{self.source_lang}' not supported")
        if self.target_lang not in DEEPL_LANG_CODES:
            raise ValueError(f"Target language '{self.target_lang}' not supported")
        
        # Initialize paths
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data"
        self.source_dir = self.data_dir / self.source_lang
        self.target_dir = self.data_dir / self.target_lang
        self.unified_dir = self.data_dir / "all-texts-unified"
        
        # Initialize DeepL client if not dry run
        if not dry_run:
            auth_key = os.getenv('DEEPL_AUTH_KEY')
            if not auth_key:
                raise ValueError("DEEPL_AUTH_KEY environment variable not set")
            self.translator = deepl.Translator(auth_key)
        else:
            self.translator = None
            
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.project_root / f"translation_{self.source_lang}_{self.target_lang}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_annotations(self, lang: str) -> Dict[str, TextAnnotation]:
        """Load annotations from subtask-2-annotations.txt file."""
        annotations = {}
        annotations_file = self.data_dir / lang / "subtask-2-annotations.txt"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
            
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) != 3:
                    self.logger.warning(f"Skipping malformed annotation line: {line}")
                    continue
                    
                file_id, narratives, subnarratives = parts
                annotations[file_id] = TextAnnotation(
                    file_id=file_id,
                    narratives=narratives,
                    subnarratives=subnarratives
                )
                
        self.logger.info(f"Loaded {len(annotations)} annotations for {lang}")
        return annotations
        
    def load_existing_translations(self) -> Set[str]:
        """Load already translated file IDs from unified annotations."""
        existing = set()
        unified_annotations = self.unified_dir / "unified-annotations.tsv"
        
        if not unified_annotations.exists():
            self.logger.info("No existing unified annotations found")
            return existing
            
        with open(unified_annotations, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row:
                    file_id = row[0]
                    # Check if this is a translation of our source language
                    if file_id.startswith(f"{self.source_lang}_TRANS_"):
                        existing.add(file_id)
                        
        self.logger.info(f"Found {len(existing)} existing translations")
        return existing
        
    def get_translation_jobs(self) -> List[TranslationJob]:
        """Identify texts that need translation."""
        source_annotations = self.load_annotations(self.source_lang)
        existing_translations = self.load_existing_translations()
        
        jobs = []
        source_texts_dir = self.source_dir / "raw-documents"
        
        if not source_texts_dir.exists():
            raise FileNotFoundError(f"Source texts directory not found: {source_texts_dir}")
            
        for file_id, annotation in source_annotations.items():
            # Generate translated file ID
            translated_file_id = f"{self.source_lang}_TRANS_{self.target_lang}_{file_id}"
            
            # Skip if already translated
            if translated_file_id in existing_translations:
                self.logger.debug(f"Skipping already translated: {file_id}")
                continue
                
            # Check if source text file exists
            source_file = source_texts_dir / file_id
            if not source_file.exists():
                self.logger.warning(f"Source text file not found: {source_file}")
                continue
                
            # Read source text
            with open(source_file, 'r', encoding='utf-8') as f:
                original_text = f.read().strip()
                
            if not original_text:
                self.logger.warning(f"Empty source text file: {source_file}")
                continue
                
            # Create target file path in unified texts directory
            target_file_path = self.unified_dir / "texts" / translated_file_id
            
            jobs.append(TranslationJob(
                file_id=translated_file_id,
                original_text=original_text,
                annotation=annotation,
                target_file_path=target_file_path
            ))
            
        self.logger.info(f"Found {len(jobs)} texts to translate")
        return jobs
        
    def translate_text(self, text: str) -> str:
        """Translate a single text using DeepL API."""
        if self.dry_run:
            return f"[DRY RUN] Translated: {text[:100]}..."
            
        try:
            result = self.translator.translate_text(
                text,
                target_lang=DEEPL_LANG_CODES[self.target_lang],
                source_lang=DEEPL_LANG_CODES[self.source_lang]
            )
            return result.text
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            raise
            
    def process_translations(self, jobs: List[TranslationJob]) -> List[TranslationJob]:
        """Process translation jobs in batches."""
        if not jobs:
            self.logger.info("No translation jobs to process")
            return []
            
        completed_jobs = []
        
        # Ensure target directories exist
        self.unified_dir.mkdir(exist_ok=True)
        (self.unified_dir / "texts").mkdir(exist_ok=True)
        
        self.logger.info(f"Starting translation of {len(jobs)} texts...")
        
        for i in range(0, len(jobs), self.batch_size):
            batch = jobs[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1} ({len(batch)} texts)")
            
            for job in batch:
                try:
                    # Translate text
                    translated_text = self.translate_text(job.original_text)
                    
                    # Write translated text to file
                    if not self.dry_run:
                        with open(job.target_file_path, 'w', encoding='utf-8') as f:
                            f.write(translated_text)
                    
                    self.logger.info(f"Translated: {job.file_id}")
                    completed_jobs.append(job)
                    
                    # Add small delay to respect API rate limits
                    if not self.dry_run:
                        time.sleep(0.1)
                        
                except Exception as e:
                    self.logger.error(f"Failed to translate {job.file_id}: {e}")
                    
            # Add delay between batches
            if not self.dry_run and i + self.batch_size < len(jobs):
                self.logger.info("Sleeping between batches...")
                time.sleep(2)
                
        self.logger.info(f"Completed {len(completed_jobs)} translations")
        return completed_jobs
        
    def update_unified_annotations(self, completed_jobs: List[TranslationJob]):
        """Update the unified annotations file with new translations."""
        if not completed_jobs:
            self.logger.info("No completed jobs to add to annotations")
            return
            
        unified_annotations = self.unified_dir / "unified-annotations.tsv"
        
        # Read existing annotations
        existing_rows = []
        if unified_annotations.exists():
            with open(unified_annotations, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                existing_rows = list(reader)
                
        # Add new annotations
        new_rows = []
        for job in completed_jobs:
            new_row = [job.file_id, job.annotation.narratives, job.annotation.subnarratives]
            new_rows.append(new_row)
            
        # Write all annotations back
        if not self.dry_run:
            all_rows = existing_rows + new_rows
            with open(unified_annotations, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(all_rows)
                
        self.logger.info(f"Added {len(new_rows)} annotations to unified file")
        
    def update_dataset_summary(self):
        """Update the dataset summary with new counts."""
        if self.dry_run:
            self.logger.info("Dry run: skipping dataset summary update")
            return
            
        summary_file = self.unified_dir / "dataset-summary.txt"
        
        # Count files in texts directory
        texts_dir = self.unified_dir / "texts"
        if not texts_dir.exists():
            return
            
        total_files = len(list(texts_dir.glob("*.txt")))
        
        # Count by language and type
        lang_counts = {}
        type_counts = {"original": 0, "translated": 0}
        
        for file_path in texts_dir.glob("*.txt"):
            file_id = file_path.stem
            
            if "_TRANS_" in file_id:
                type_counts["translated"] += 1
                # Extract target language from translated file ID
                parts = file_id.split("_")
                if len(parts) >= 3:
                    target_lang = parts[2]
                    lang_counts[target_lang] = lang_counts.get(target_lang, 0) + 1
            else:
                type_counts["original"] += 1
                # Extract language from original file ID
                if file_id.startswith(("EN_", "PT_", "RU_", "BG_", "HI_")):
                    lang = file_id.split("_")[0]
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    
        # Write summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("UNIFIED DATASET SUMMARY\n")
            f.write("=====================\n\n")
            f.write(f"Total files: {total_files}\n\n")
            f.write("By Language:\n")
            for lang in sorted(lang_counts.keys()):
                f.write(f"  {lang}: {lang_counts[lang]}\n")
            f.write("\nBy Type:\n")
            for type_name in sorted(type_counts.keys()):
                f.write(f"  {type_name}: {type_counts[type_name]}\n")
                
        self.logger.info("Updated dataset summary")
        
    def run(self):
        """Execute the complete translation pipeline."""
        self.logger.info(f"Starting translation: {LANG_NAMES[self.source_lang]} -> {LANG_NAMES[self.target_lang]}")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual translations or file changes will be made")
            
        # Get translation jobs
        jobs = self.get_translation_jobs()
        
        if not jobs:
            self.logger.info("No new translations needed")
            return
            
        # Process translations
        completed_jobs = self.process_translations(jobs)
        
        # Update unified annotations
        self.update_unified_annotations(completed_jobs)
        
        # Update dataset summary
        self.update_dataset_summary()
        
        self.logger.info(f"Translation pipeline completed. Processed {len(completed_jobs)} texts.")


def main():
    parser = argparse.ArgumentParser(
        description="Translate texts using DeepL API for hybrid text classification project"
    )
    parser.add_argument("--source", required=True, choices=list(DEEPL_LANG_CODES.keys()),
                        help="Source language code")
    parser.add_argument("--target", required=True, choices=list(DEEPL_LANG_CODES.keys()),
                        help="Target language code") 
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of texts to process per batch (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run in dry-run mode (no actual translations or file changes)")
    
    args = parser.parse_args()
    
    if args.source == args.target:
        print("Error: Source and target languages cannot be the same")
        sys.exit(1)
        
    try:
        translator = DeepLTranslator(
            source_lang=args.source,
            target_lang=args.target,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        translator.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()