import click
import json
import os
from pathlib import Path

def validate_json_structure(content):
    """Validate the JSON can be parsed."""
    return True  # If we got here, json.load() succeeded

@click.group()
def cli():
    """CLI tool for validating and uploading model submissions."""
    pass

@cli.command()
@click.argument('task_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
def validate(task_dir, output_dir):
    """Validate output files against task directory."""
    click.echo(f"Starting validation...")
    
    task_path = Path(task_dir)
    output_path = Path(output_dir)
    
    click.echo(f"Task directory: {task_path.absolute()}")
    click.echo(f"Output directory: {output_path.absolute()}")
    
    # Count JSON files in task directory
    task_files = list(task_path.glob('*.json'))
    expected_count = len(task_files)
    
    # Count and validate output files
    output_files = list(output_path.glob('*.json'))
    
    click.echo(f"\nValidating {output_path.name}...")
    click.echo(f"Found {len(output_files)} files (expected {expected_count})")
    
    
    if len(output_files) != expected_count:
        click.echo(f"❌ Error: Output directory must contain exactly {expected_count} files (found in {task_dir})")
        is_valid = False
    else:
        click.echo(f"✅ Output directory contains {expected_count} files")
        is_valid = True
    
    invalid_files = []
    for file in output_files:
        try:
            with open(file) as f:
                json.load(f)
        except json.JSONDecodeError:
            invalid_files.append(file.name)
            
    if invalid_files:
        click.echo("\n❌ The following files have invalid JSON:")
        for file in invalid_files:
            click.echo(f"  - {file}")
        is_valid = False
    else:
        click.echo("\n✅ All files contain valid JSON")

    return is_valid

@cli.command()
@click.argument('model_dir', type=click.Path(exists=True))
def upload(model_dir):
    """Upload JSON files to Hugging Face dataset."""
    click.echo(f"Starting upload...")
    
    path = Path(model_dir)
    model_name = path.name
    
    api = HfApi()
    repo_id = f"renee-evals/{model_name}"
    
    try:
        # Create or get dataset repository
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        
        # Upload all JSON files
        files = list(path.glob('*.json'))
        click.echo(f"Uploading {len(files)} files to {repo_id}...")
        
        for file in files:
            api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=file.name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            click.echo(f"Uploaded {file.name}")
            
        click.echo(f"\n✅ Successfully uploaded all files to {repo_id}")
        return True
        
    except Exception as e:
        click.echo(f"\n❌ Upload failed: {str(e)}")
        return False

if __name__ == '__main__':
    cli() 