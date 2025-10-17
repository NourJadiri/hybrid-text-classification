#!/usr/bin/env python3
"""
Script to automatically submit result files to SemEval 2025 Task 10 evaluation server
and retrieve the evaluation results.
"""

import requests
import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
import time

# Configuration
BASE_URL = "https://propaganda.math.unipd.it/semeval2025task10"
TEAM_PAGE_URL = f"{BASE_URL}/teampage.php?passcode=33919dec6bb13d9ba70150aca9519aac"
UPLOAD_URL = f"{BASE_URL}/upload.php"
PASSCODE = "33919dec6bb13d9ba70150aca9519aac"

def submit_file(file_path: Path, session: requests.Session, output_dir: Path):
    """
    Submit a single file to the evaluation server.
    
    Args:
        file_path: Path to the result file
        session: Requests session to maintain cookies
        output_dir: Directory to save debug output
        
    Returns:
        tuple: (success: bool, result_text: str)
    """
    print(f"\n{'='*70}")
    print(f"Submitting: {file_path.name}")
    print(f"{'='*70}")
    
    try:
        # First, fetch the upload page to ensure we have all cookies and CSRF tokens
        print("Fetching upload page to get session cookies...")
        page_response = session.get(UPLOAD_URL, timeout=10)
        soup = BeautifulSoup(page_response.text, 'html.parser')
        
        # Look for any hidden CSRF tokens or similar
        csrf_token = None
        for hidden_input in soup.find_all('input', type='hidden'):
            if hidden_input.get('name') in ['csrf', 'token', '_token']:
                csrf_value = hidden_input.get('value')
                if isinstance(csrf_value, str):
                    csrf_token = csrf_value
                elif isinstance(csrf_value, list):
                    csrf_token = csrf_value[0] if csrf_value else None
                break
        
        # Open the file
        with open(file_path, 'rb') as f:
            files = {
                'fileToUpload': (file_path.name, f, 'text/plain')
            }
            
            # Prepare form data
            data = {
                'passcode': PASSCODE
            }
            
            # Add CSRF token if found
            if csrf_token:
                data['csrf'] = str(csrf_token)
                print(f"Found CSRF token, including in submission")
            
            # Set browser-like headers matching your exact browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Origin': 'https://propaganda.math.unipd.it',
                'Referer': f'{BASE_URL}/teampage.php?passcode={PASSCODE}',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-User': '?1',
                'Sec-GPC': '1',
                'Upgrade-Insecure-Requests': '1',
                'sec-ch-ua': '"Brave";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }
            
            # Submit the form
            print("Uploading file to server...")
            response = session.post(
                UPLOAD_URL,
                files=files,
                data=data,
                headers=headers,
                timeout=60,
                allow_redirects=True
            )
            
            # Check if successful
            if response.status_code == 200:
                print("✓ Upload successful!")
                
                # Parse the response HTML to extract results
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Save full HTML response for debugging in output directory
                debug_file = output_dir / f"{file_path.stem}_response.html"
                with open(debug_file, 'w', encoding='utf-8') as df:
                    df.write(response.text)
                print(f"✓ Full response saved to: {debug_file}")
                
                # Try to extract evaluation results
                # (This part may need adjustment based on actual server response format)
                result_text = response.text
                
                # Look for common result indicators
                if "error" in result_text.lower() and "upload" not in result_text.lower():
                    print("✗ Server returned an error")
                    return False, result_text
                
                print("✓ Evaluation complete!")
                return True, result_text
                
            else:
                print(f"✗ Upload failed with status code: {response.status_code}")
                return False, f"HTTP Error {response.status_code}"
                
    except Exception as e:
        print(f"✗ Error during submission: {e}")
        return False, str(e)

def process_results_folder(folder_path: str, output_dir: str = "evaluation_results"):
    """
    Process all .txt files in a folder and submit them for evaluation.
    
    Args:
        folder_path: Path to folder containing result files
        output_dir: Directory to save evaluation results
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all .txt files
    txt_files = sorted(folder.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return
    
    print(f"Found {len(txt_files)} result files to submit")
    print(f"Results will be saved to: {output_path}")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Add the visitor tracking cookie
    session.cookies.set('sc_is_visitor_unique', 'rx11864198.1760649262.F0BE9B4DBF304992B60F5D9C300BBF0F.40.32.27.22.19.14.8.6.1')
    
    # First, visit the team page to establish session
    print("\nEstablishing session with server...")
    try:
        session.get(TEAM_PAGE_URL, timeout=10)
        print("✓ Session established")
        print("✓ Visitor tracking cookie added")
    except Exception as e:
        print(f"✗ Failed to establish session: {e}")
        return
    
    # Process each file
    results_summary = []
    
    for i, file_path in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] Processing {file_path.name}...")
        
        success, result = submit_file(file_path, session, output_path)
        
        results_summary.append({
            'file': file_path.name,
            'success': success,
            'result': result
        })
        
        # Save individual result
        result_file = output_path / f"{file_path.stem}_evaluation.html"
        with open(result_file, 'w', encoding='utf-8') as rf:
            rf.write(result)
        print(f"✓ Result saved to: {result_file}")
        
        # Be polite to the server - wait between submissions
        if i < len(txt_files):
            wait_time = 2
            print(f"Waiting {wait_time} seconds before next submission...")
            time.sleep(wait_time)
    
    # Print summary
    print("\n" + "="*70)
    print("SUBMISSION SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results_summary if r['success'])
    print(f"\nTotal files: {len(results_summary)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results_summary) - successful}")
    
    print("\nDetailed results:")
    for result in results_summary:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['file']}")
    
    # Save summary to file
    summary_file = output_path / "submission_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as sf:
        sf.write("SemEval 2025 Task 10 - Submission Summary\n")
        sf.write("=" * 70 + "\n\n")
        sf.write(f"Total files: {len(results_summary)}\n")
        sf.write(f"Successful: {successful}\n")
        sf.write(f"Failed: {len(results_summary) - successful}\n\n")
        sf.write("Detailed results:\n")
        for result in results_summary:
            status = "SUCCESS" if result['success'] else "FAILED"
            sf.write(f"  [{status}] {result['file']}\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Submit result files to SemEval 2025 Task 10 evaluation server"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="results/openai_gpt-5-nano",
        help="Folder containing result .txt files (default: results/openai_gpt-5-nano)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Submit a single file instead of entire folder"
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Submit single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        session = requests.Session()
        session.cookies.set('sc_is_visitor_unique', 'rx11864198.1760649262.F0BE9B4DBF304992B60F5D9C300BBF0F.40.32.27.22.19.14.8.6.1')
        session.get(TEAM_PAGE_URL, timeout=10)
        
        success, result = submit_file(file_path, session, output_path)
        
        result_file = output_path / f"{file_path.stem}_evaluation.html"
        with open(result_file, 'w', encoding='utf-8') as rf:
            rf.write(result)
        
        print(f"\nResult saved to: {result_file}")
        
    else:
        # Process entire folder
        process_results_folder(args.folder, args.output)

if __name__ == "__main__":
    main()
