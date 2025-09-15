"""
Main script to run the classification system with configuration files.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from config_loader import ClassificationConfig, create_example_config
from graph_builder import build_graph_from_config
from utils import get_unprocessed_texts
from state import ClassificationState

load_dotenv()

async def run_classification(config_path: str) -> None:
    """
    Run the classification system with a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    print(f"[Main] Loading configuration from: {config_path}")
    
    try:
        print(f"[Main] Building graph with configuration...")
        graph, execution_config, config = build_graph_from_config(config_path)
        print(f"[Main] Graph compiled successfully")
        
        print(f"[Main] Loading texts from: {config.input_folder}")
        text_list, file_names = get_unprocessed_texts(config.input_folder, config.output_file)
        
        if not text_list:
            print(f"[Main] No unprocessed text files found in {config.input_folder}")
            if Path(config.output_file).exists():
                print(f"[Main] All files appear to have been processed already")
            else:
                print(f"[Main] No text files found at all in the input folder")
            return
        
        initial_states_batch = [
            {"text": text, "file_id": file_id} 
            for text, file_id in zip(text_list, file_names)
        ]
        
        print(f"[Main] Processing {len(initial_states_batch)} files...")
        print(f"[Main] Output will be appended to: {config.output_file}")
        
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Type cast for compatibility - runtime behavior is correct
        typed_states = cast(list[ClassificationState], initial_states_batch)
        await graph.abatch(typed_states, config=execution_config)  # type: ignore[arg-type]
        
        print(f"[Main] Classification completed successfully!")
        print(f"[Main] Results saved to: {config.output_file}")
        
    except Exception as e:
        print(f"[Main] Error during classification: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run text classification with configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_with_config.py --create-example-config my_config.yaml
  python run_with_config.py --config my_config.yaml
  python run_with_config.py --config experiment1.yaml experiment2.yaml experiment3.yaml
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", 
        type=str, 
        nargs="+",
        help="Path(s) to YAML configuration file(s)"
    )
    group.add_argument(
        "--create-example-config", 
        type=str, 
        metavar="OUTPUT_PATH",
        help="Create an example configuration file at the specified path"
    )
    
    args = parser.parse_args()
    
    if args.create_example_config:
        create_example_config(args.create_example_config)
        print(f"Example configuration created at: {args.create_example_config}")
        print("Edit the configuration file and run with --config to start classification.")
        return
    
    for config_path in args.config:
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"[Error] Configuration file not found: {config_path}")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Running experiment with config: {config_path.name}")
        print(f"{'='*60}")
        
        try:
            asyncio.run(run_classification(str(config_path)))
        except KeyboardInterrupt:
            print(f"\n[Main] Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"[Error] Failed to run experiment with {config_path}: {e}")
            continue
        
        print(f"{'='*60}")
        print(f"Completed experiment: {config_path.name}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
