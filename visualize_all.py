import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_arc_colormap():
    """Return the ARC app color palette as a ListedColormap."""
    # ARC app colors (0-9)
    colors = [
        '#000000',  # 0: black
        '#0074D9',  # 1: blue
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: light gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: light blue
        '#870C25'   # 9: dark red
    ]
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)

def plot_grid(ax, grid, title, is_prediction=False):
    if not grid or not any(grid):
        ax.text(0.5, 0.5, 'Empty', ha='center', va='center')
        ax.set_title(f'{title}\n(Empty)', fontsize=10)
        return
        
    grid_array = np.array(grid)
    cmap = get_arc_colormap()
    
    # For predictions, add a border to distinguish
    if is_prediction:
        from matplotlib.patches import Rectangle
        rect = Rectangle((-0.5, -0.5), grid_array.shape[1], grid_array.shape[0], 
                        linewidth=3, edgecolor='#2ECC40', facecolor='none', linestyle='--')
        ax.add_patch(rect)
    
    im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Add text annotations (only for non-zero cells)
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            if grid_array[i, j] > 0:  # Only show text for non-zero cells
                ax.text(j, i, str(grid_array[i, j]), 
                       ha='center', va='center',
                       color='white' if grid_array[i, j] in [1, 2, 3, 4, 6, 7, 8, 9] else 'black',
                       fontsize=14, fontweight='bold')

def visualize_all(task_id: str, data_dir: str, submission_dir: str, output_dir: str = 'visualizations'):
    """Visualize all training examples, test inputs, and model predictions with ARC app styling."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style to look more like the ARC app
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.facecolor'] = '#ffffff'
    
    # Load task data
    task_file = Path(data_dir) / f"{task_id}.json"
    task_data = load_json(task_file)
    
    # Load submission
    submission_file = Path(submission_dir) / f"{task_id}.json"
    submission = load_json(submission_file)
    
    # Get all training and test examples
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    
    # Create a figure with subplots for all examples
    n_train = len(train_examples)
    n_test = len(test_examples)
    n_cols = 3  # Input, Expected Output, Prediction
    n_rows = max(n_train, 1) + max(n_test, 1)  # At least one row for each section
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(15, 5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.3)
    
    # Plot training examples
    if n_train > 0:
        # Add section title for training examples
        fig.text(0.5, 1.0 - 0.5/n_rows, 'Training Examples', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        for i, example in enumerate(train_examples):
            # Input
            ax = fig.add_subplot(gs[i, 0])
            plot_grid(ax, example.get('input', []), f'Input {i+1}')
            
            # Expected output
            ax = fig.add_subplot(gs[i, 1])
            plot_grid(ax, example.get('output', []), f'Expected Output {i+1}')
            
            # Empty for training (no prediction)
            ax = fig.add_subplot(gs[i, 2])
            ax.axis('off')
    
    # Plot test examples and predictions
    if n_test > 0:
        start_row = max(n_train, 1)  # Start after training examples or at row 1 if no training examples
        
        # Add section title for test examples
        fig.text(0.5, 1.0 - (start_row + 0.5)/n_rows, 'Test Examples', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        for i, test in enumerate(test_examples):
            row = start_row + i
            if row >= n_rows:  # Safety check
                break
                
            # Test input
            ax = fig.add_subplot(gs[row, 0])
            plot_grid(ax, test.get('input', []), f'Test Input {i+1}')
            
            # Expected output (if available)
            ax = fig.add_subplot(gs[row, 1])
            expected_output = test.get('output', [])
            if not expected_output:  # If no expected output, try to get it from the task data
                expected_output = test.get('expected_output', [])
            plot_grid(ax, expected_output, f'Expected Output {i+1}')
            
            # Model prediction
            ax = fig.add_subplot(gs[row, 2])
            if i < len(submission):
                # Get the last attempt's answer
                attempts = [k for k in submission[i].keys() if k.startswith('attempt_')]
                if attempts:
                    last_attempt = max(attempts)
                    plot_grid(ax, 
                             submission[i][last_attempt].get('answer', []), 
                             f'Model Prediction {i+1}',
                             is_prediction=True)
                else:
                    ax.text(0.5, 0.5, 'No prediction', ha='center', va='center')
                    ax.set_title(f'Model Prediction {i+1}\n(No prediction)', fontsize=10)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No submission', ha='center', va='center')
                ax.set_title(f'Model Prediction {i+1}\n(No submission)', fontsize=10)
                ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = Path(output_dir) / f'{task_id}_all.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize all training and test examples with model predictions')
    parser.add_argument('--task_id', type=str, default='f0afb749', help='Task ID to visualize')
    parser.add_argument('--data_dir', type=str, default='data/arc-agi/data/evaluation', help='Directory containing task data')
    parser.add_argument('--submission_dir', type=str, default='submissions/gpt5-mini-minimal', help='Directory containing model submissions')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    visualize_all(args.task_id, args.data_dir, args.submission_dir, args.output_dir)
