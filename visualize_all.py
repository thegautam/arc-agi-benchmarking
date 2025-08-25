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
    ax.clear()
    
    # Handle None or empty grid
    if grid is None:
        ax.text(0.5, 0.5, 'None', ha='center', va='center')
        ax.set_title(f'{title}\n(None)', fontsize=10)
        ax.axis('off')
        return
        
    if isinstance(grid, (list, tuple)) and not grid:
        ax.text(0.5, 0.5, 'Empty', ha='center', va='center')
        ax.set_title(f'{title}\n(Empty)', fontsize=10)
        ax.axis('off')
        return
    
    try:
        # Convert to 2D numpy array, handling various input formats
        if isinstance(grid, (int, float)):
            grid_array = np.array([[grid]], dtype=int)
        elif isinstance(grid, (list, tuple)):
            # Handle single number in a list
            if all(isinstance(x, (int, float)) for x in grid):
                grid_array = np.array([grid], dtype=int)
            # Handle list of lists
            elif all(isinstance(row, (list, tuple)) for row in grid):
                # Find max row length for padding
                max_len = max(len(row) for row in grid) if grid else 0
                # Pad rows to make them the same length
                padded_grid = []
                for row in grid:
                    if not row:  # Handle empty rows
                        padded_grid.append([0] * max_len)
                    else:
                        padded_row = []
                        for val in row:
                            # Convert any non-numeric values to 0
                            try:
                                padded_row.append(int(float(val)) if val is not None else 0)
                            except (ValueError, TypeError):
                                padded_row.append(0)
                        # Pad the row if needed
                        padded_row.extend([0] * (max_len - len(padded_row)))
                        padded_grid.append(padded_row)
                grid_array = np.array(padded_grid, dtype=int)
            else:
                raise ValueError("Inconsistent grid structure")
        else:
            # Try to convert directly to numpy array
            grid_array = np.array(grid, dtype=int)
        
        # Ensure 2D array
        if grid_array.ndim == 0:  # Single number
            grid_array = np.array([[grid_array]])
        elif grid_array.ndim == 1:  # Single row
            grid_array = np.array([grid_array])
        elif grid_array.ndim > 2:
            grid_array = grid_array.reshape(grid_array.shape[0], -1)  # Flatten extra dimensions
        
        # Get color map
        cmap = get_arc_colormap()
        
        # For predictions, add a border to distinguish
        if is_prediction and grid_array.size > 0:
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (-0.5, -0.5), 
                grid_array.shape[1], 
                grid_array.shape[0], 
                linewidth=3, 
                edgecolor='#2ECC40', 
                facecolor='none', 
                linestyle='--'
            )
            ax.add_patch(rect)
        
        # Plot the grid if not empty
        if grid_array.size > 0:
            # Replace any negative values with 0 for display
            grid_array = np.maximum(grid_array, 0)
            # Cap values at 9 for the color map
            grid_array = np.minimum(grid_array, 9)
            
            im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
            
            # Add text annotations for non-zero cells
            for i in range(grid_array.shape[0]):
                for j in range(grid_array.shape[1]):
                    val = grid_array[i, j]
                    if val > 0:  # Only show text for non-zero cells
                        ax.text(
                            j, i, 
                            str(int(val)), 
                            ha='center', 
                            va='center',
                            color='white' if val in [1, 2, 3, 4, 6, 7, 8, 9] else 'black',
                            fontsize=min(14, 100/max(grid_array.shape)),  # Scale font with grid size
                            fontweight='bold'
                        )
        
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines if we have a grid
        if grid_array.size > 0:
            ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
    except Exception as e:
        # Fallback for any errors
        ax.clear()
        ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        error_msg = str(e)
        ax.set_title(f'{title}\n({error_msg[:20]}...)', fontsize=8)
        ax.axis('off')
        # Log the full error for debugging
        import traceback
        print(f"Error plotting {title}:")
        print(traceback.format_exc())

def visualize_all(task_id: str, data_dir: str, submission_dir: str, output_dir: str = 'visualizations'):
    """Visualize all training examples, test inputs, and model predictions with ARC app styling."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style to look more like the ARC app
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['figure.autolayout'] = False  # Disable autolayout to prevent tight_layout warnings
    
    # Load task data
    task_file = Path(data_dir) / f"{task_id}.json"
    task_data = load_json(task_file)
    
    # Load submission
    submission_file = Path(submission_dir) / f"{task_id}.json"
    submission = load_json(submission_file)
    
    # Get all training and test examples
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    
    # Calculate grid layout
    n_train = len(train_examples)
    n_test = len(test_examples)
    n_cols = 3  # Input, Expected Output, Prediction
    
    # Calculate number of rows needed:
    # 1 row for column headers
    # 1 row for training header + max(n_train, 1) for training examples
    # 1 row for test header + max(n_test, 1) for test examples
    n_rows = 1 + 1 + max(n_train, 1) + 1 + max(n_test, 1)
    
    # Create figure with appropriate size, with more height per row for better spacing
    fig = plt.figure(figsize=(15, 4 * n_rows + 2))
    
    # Create a grid with space for headers and content
    # First row: column headers
    # Next n_train rows: training examples
    # Next row: test header
    # Remaining rows: test examples
    height_ratios = [0.5] + [1] * (n_rows - 1)  # Header is half height
    
    gs = fig.add_gridspec(
        n_rows,  
        n_cols, 
        height_ratios=height_ratios,
        hspace=0.5, 
        wspace=0.3
    )
    
    # Add section headers
    header_style = {'fontsize': 14, 'fontweight': 'bold', 'ha': 'center', 'va': 'center', 'color': '#333333'}
    
    # Column headers
    for col, title in enumerate(['Input', 'Expected Output', 'Prediction']):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, title, **header_style)
        ax.axis('off')
    
    # Plot training examples
    current_row = 1  # Start after column headers
    
    if n_train > 0:
        # Add section title for training examples
        ax = fig.add_subplot(gs[current_row, :])
        ax.text(0.5, 0.5, 'Training Examples', **header_style)
        ax.axis('off')
        current_row += 1
        
        for i, example in enumerate(train_examples):
            # Input
            ax = fig.add_subplot(gs[current_row, 0])
            plot_grid(ax, example.get('input', []), f'Example {i+1} Input')
            
            # Expected output
            ax = fig.add_subplot(gs[current_row, 1])
            plot_grid(ax, example.get('output', []), f'Example {i+1} Expected')
            
            # Empty for training (no prediction)
            ax = fig.add_subplot(gs[current_row, 2])
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
            ax.set_title(f'Example {i+1} Prediction', fontsize=10)
            ax.axis('off')
            
            current_row += 1
    
    # Plot test examples and predictions
    if n_test > 0:
        # Add section title for test examples
        ax = fig.add_subplot(gs[current_row, :])
        ax.text(0.5, 0.5, 'Test Examples', **header_style)
        ax.axis('off')
        current_row += 1
        
        for i, example in enumerate(test_examples):
            # Input
            ax = fig.add_subplot(gs[current_row, 0])
            plot_grid(ax, example.get('input', []), f'Test {i+1} Input')
            
            # Expected output (if available)
            ax = fig.add_subplot(gs[current_row, 1])
            plot_grid(ax, example.get('output', []), f'Test {i+1} Expected')
            
            # Model prediction
            ax = fig.add_subplot(gs[current_row, 2])
            prediction = None
            
            if not submission:
                # No submission at all
                ax.text(0.5, 0.5, 'No submission', ha='center', va='center', color='gray')
                ax.set_title(f'Test {i+1} Prediction\n(No submission)', fontsize=10)
            elif isinstance(submission, list) and i < len(submission):
                if isinstance(submission[i], dict):
                    # Handle submission with attempt information
                    attempts = [k for k in submission[i].keys() if k.startswith('attempt_')]
                    if attempts:
                        last_attempt = max(attempts)
                        prediction = submission[i][last_attempt].get('answer', None)
                        plot_grid(ax, prediction, f'Test {i+1} Prediction', is_prediction=True)
                    else:
                        # No attempts found
                        ax.text(0.5, 0.5, 'No prediction', ha='center', va='center', color='gray')
                        ax.set_title(f'Test {i+1} Prediction\n(No attempts)', fontsize=10)
                else:
                    # Simple list of outputs
                    prediction = submission[i].get('output', None) if isinstance(submission[i], dict) else submission[i]
                    plot_grid(ax, prediction, f'Test {i+1} Prediction', is_prediction=True)
            else:
                # Unsupported submission format
                ax.text(0.5, 0.5, 'Invalid format', ha='center', va='center', color='gray')
                ax.set_title(f'Test {i+1} Prediction\n(Invalid format)', fontsize=10)
            
            ax.axis('off')
            current_row += 1  # Move to next row for next example
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = Path(output_dir) / f'{task_id}_all.png'
    
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    print(f"Saving visualization to: {output_path.absolute()}")
    try:
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
        print(f"Successfully saved visualization to: {output_path.absolute()}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        # Try saving to current directory as fallback
        fallback_path = Path(f'{task_id}_all.png')
        try:
            plt.savefig(str(fallback_path), bbox_inches='tight', dpi=150)
            print(f"Saved visualization to fallback location: {fallback_path.absolute()}")
        except Exception as e2:
            print(f"Failed to save visualization: {e2}")
    
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize all training and test examples with model predictions')
    parser.add_argument('--task_id', type=str, default='f0afb749', help='Task ID to visualize')
    parser.add_argument('--data_dir', type=str, default='data/arc-agi/data/evaluation', help='Directory containing task data')
    parser.add_argument('--submission_dir', type=str, default='submissions/gpt5-mini-minimal', help='Directory containing model submissions')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    visualize_all(args.task_id, args.data_dir, args.submission_dir, args.output_dir)
