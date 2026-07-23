import subprocess
import sys
import os
import json
import requests
import time
import argparse

def check_ollama():
    print("Checking Ollama status...")
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]
            print(f"Ollama is running. Available models: {model_names}")
            
            # Check for mistral model (either mistral, mistral:latest, etc.)
            has_mistral = any('mistral' in name.lower() for name in model_names)
            if not has_mistral:
                print("WARNING: 'mistral' model not found in Ollama. Pulling 'mistral' model now...")
                pull_response = requests.post('http://localhost:11434/api/pull', json={"name": "mistral"}, stream=True)
                for line in pull_response.iter_lines():
                    if line:
                        print(line.decode('utf-8'))
            return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama is not running on http://localhost:11434.")
        print("Please start the Ollama application on your computer first, then run this script.")
        return False
    return False

def run_command(command, description):
    print(f"\n==========================================")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"==========================================")
    
    start_time = time.time()
    try:
        # Run subprocess and stream output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        duration = time.time() - start_time
        
        if process.returncode == 0:
            print(f"SUCCESS: {description} finished in {duration:.2f} seconds.")
            return True
        else:
            print(f"FAILED: {description} exited with code {process.returncode}.")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return False

def main():
    print("=== STARTING SIGEXT WORKING PIPELINE ===")
    
    parser = argparse.ArgumentParser(description="Run complete SigExt pipeline.")
    parser.add_argument("--tune_threshold", action="store_true", help="Tune threshold and top-k on validation set.")
    parser.add_argument("--dedup_strategy", default="fuzzy", choices=["exact", "substring", "fuzzy"], help="Keyword deduplication strategy.")
    parser.add_argument("--dedup_threshold", default=70, type=int, help="Similarity threshold for fuzzy deduplication.")
    
    args = parser.parse_args()
    
    # 1. Verify Ollama is running
    if not check_ollama():
        sys.exit(1)
        
    # Define directories
    dataset_dir = "experiments/test_dataset/"
    checkpoint_dir = "experiments/test_checkpoint/"
    keyphrase_output_dir = "experiments/test_dataset_with_keyphrase_new/"
    summary_output_dir = "experiments/cnn_extsig_predictions_test/"
    
    # 2. Run Inference Step
    inference_cmd = [
        sys.executable,
        "src/inference_longformer_extractor.py",
        "--dataset_dir", dataset_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", keyphrase_output_dir
    ]
    
    success = run_command(inference_cmd, "Stage 1: Keyphrase Extraction (Inference)")
    if not success:
        sys.exit(1)
        
    # 3. Run Summarization Step
    summarization_cmd = [
        sys.executable,
        "src/zs_summarization.py",
        "--model_name", "mistral",
        "--kw_strategy", "sigext_topk",
        "--kw_model_top_k", "15",
        "--dataset", "cnn",
        "--dataset_dir", keyphrase_output_dir,
        "--output_dir", summary_output_dir,
        "--dedup_strategy", args.dedup_strategy,
        "--dedup_threshold", str(args.dedup_threshold)
    ]
    if args.tune_threshold:
        summarization_cmd.append("--tune_threshold")
        
    success = run_command(summarization_cmd, "Stage 2: Zero-Shot Summarization & Evaluation")
    if not success:
        sys.exit(1)
        
    # 4. Display Results
    metrics_file = os.path.join(summary_output_dir, "test_metrics.json")
    if os.path.exists(metrics_file):
        print(f"\n==========================================")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("Final Evaluation ROUGE Scores:")
        print(f"==========================================")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            print(json.dumps(metrics, indent=4))
        print(f"==========================================")
    else:
        print("\nERROR: Metrics file was not generated.")

if __name__ == "__main__":
    main()
