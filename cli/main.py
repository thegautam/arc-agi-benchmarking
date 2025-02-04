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
@click.argument('model_dir', type=click.Path(exists=True))
def validate(model_dir):
    """Validate JSON files in the model directory."""
    click.echo(f"Starting validation...")
    
    path = Path(model_dir)
    click.echo(f"Checking directory: {path.absolute()}")
    
    files = list(path.glob('*.json'))
    
    click.echo(f"\nValidating {path.name}...")
    click.echo(f"Found {len(files)} files (expected 100)")
    
    is_valid = True
    
    if len(files) != 100:
        click.echo("❌ Error: Directory must contain exactly 100 files")
        is_valid = False
    
    invalid_files = []
    for file in files:
        try:
            with open(file) as f:
                json.load(f)  # Just try to parse it
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

if __name__ == '__main__':
    cli() 