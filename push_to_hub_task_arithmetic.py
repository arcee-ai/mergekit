import re
import yaml
import os
import argparse
from concurrent.futures import Future
import shutil
from pathlib import Path

from huggingface_hub import (
    create_branch,
    create_repo,
    list_repo_commits,
    upload_folder,
)

def push_to_hub_revision(hub_model_id, revision, output_dir, extra_ignore_patterns=[]) -> Future:
    """Pushes the model to branch on a Hub repo."""

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=hub_model_id,
        branch=revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    print(f"Created target repo at {repo_url}")
    print(f"Pushing to the Hub revision {revision}...")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=hub_model_id,
        folder_path=output_dir,
        revision=revision,
        commit_message=f"Add {revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    print(f"Pushed to {repo_url} revision {revision} successfully!")

    return future

def extract_yaml_from_markdown(markdown_path):
    """Extract YAML configuration from a markdown file."""
    with open(markdown_path, 'r') as f:
        content = f.read()
    
    # Find YAML code blocks
    yaml_matches = re.findall(r'```yaml\s+(.*?)\s+```', content, re.DOTALL)
    
    if not yaml_matches:
        raise ValueError(f"No YAML code blocks found in {markdown_path}")
    
    # Return the first YAML block found
    return yaml_matches[0]

def extract_yaml_from_file(yaml_path):
    """Extract YAML configuration directly from a YAML file."""
    with open(yaml_path, 'r') as f:
        return f.read()

def generate_model_string(output_dir):
    """
    Extract model information from a markdown file with YAML config or directly from YAML
    and generate a formatted string.
    """
    # Join output_dir with README.md
    file_path = os.path.join(output_dir, 'README.md')
    # Determine if the file is markdown or yaml
    if file_path.endswith('.md'):
        yaml_content = extract_yaml_from_markdown(file_path)
    else:
        yaml_content = extract_yaml_from_file(file_path)
    
    config = yaml.safe_load(yaml_content)
    
    model_revs = []
    weights = []
    base_model = None
    
    for model_entry in config.get('models', []):
        # Extract the model path
        model_path = model_entry.get('model', '')
        
        # Extract the part before and after @ in the model path
        if '@' in model_path:
            model_name, revision = model_path.split('@')
            model_revs.append(revision)
            
            # Save the base model name (we only need one since they're typically the same)
            if base_model is None:
                base_model = model_name
        else:
            model_revs.append('main')  # Default if no revision specified
            if base_model is None:
                base_model = model_path
        
        # Extract the weight
        weight = model_entry.get('parameters', {}).get('weight')
        weights.append(str(weight))
    
    lambda_value = config.get('parameters', {}).get('lambda')

    # Combine into final strings
    revs_part = "_".join(model_revs)
    weights_part = "-".join(weights)
    result_string = f"{revs_part}_task-arithmetic_weights-{weights_part}_lambda-{lambda_value}"
    
    # Model name with -Merges suffix
    model_name = f"{base_model}-Merges" if base_model else "Unknown-Model-Merges"
    
    return model_name, result_string

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract model information from YAML config files")
    parser.add_argument('--path', type=str, required=True, 
                        help='Path to the YAML file or Markdown file containing YAML')
    return parser.parse_args()

def main():
    """Main function to run the script."""
    args = parse_arguments()
    output_dir = args.path
    
    try:
        model_name, result_string = generate_model_string(output_dir)
        print(f"Model name: {model_name}")
        print(f"Result string: {result_string}")
    except Exception as e:
        print(f"Error processing file {output_dir}: {e}")
        return 1

    # Push to the Hub
    future = push_to_hub_revision(model_name, result_string, output_dir)
    if future.done():
        print("Push to Hub completed successfully.")
    
    return 0

if __name__ == "__main__":
    exit(main())