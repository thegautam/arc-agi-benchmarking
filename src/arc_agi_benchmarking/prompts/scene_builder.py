from typing import List, Dict, Tuple

def get_color_counts(grid: List[List[int]]) -> Dict[int, int]:
    """Count occurrences of each color in the grid."""
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    return color_counts

def describe_grid(grid: List[List[int]], name: str = "grid") -> str:
    """Generate a description of a grid including size and color information."""
    if not grid or not grid[0]:
        return f"The {name} is empty."
    
    rows = len(grid)
    cols = len(grid[0])
    color_counts = get_color_counts(grid)
    colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    description = [
        f"{name.upper()}:",
        f"- Size: {rows} rows X {cols} columns",
        f"- Total cells: {rows * cols}",
        "- Colors (color: count):"
    ]
    
    for color, count in colors:
        description.append(f"  - {color}: {count} cells ({count / (rows * cols):.1%})")
    
    return "\n".join(description)

def build_scene_description(input_grid: List[List[int]], output_grid: List[List[int]]) -> str:
    """
    Build a scene description comparing input and output grids.
    """
    input_desc = describe_grid(input_grid, "input")
    output_desc = describe_grid(output_grid, "output")
    
    return f"SCENE DESCRIPTION\n\n{input_desc}\n\n{output_desc}"