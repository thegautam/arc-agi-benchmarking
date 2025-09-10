from typing import List, Dict, Tuple
from collections import deque

def get_color_counts(grid: List[List[int]]) -> Dict[int, int]:
    """Count occurrences of each color in the grid."""
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    return color_counts

# Get per-color bounding boxes of all non-zero components
def get_largest_bounding_box(
    grid: List[List[int]],
    blank_hops: int = 0,
) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    For each non-zero color, find components consisting of that color and return
    their bounding boxes. Connectivity is primarily 4-directional (up, down, left, right)
    through same-color cells, but can optionally traverse up to `blank_hops` consecutive
    blank (zero) cells in order to connect same-color regions.

    Returns:
        Dict[color, List[(top, left, bottom, right)]]

    Notes:
    - Connectivity is 4-directional (up, down, left, right).
    - When `blank_hops` > 0, traversal may pass through up to that many consecutive 0-cells
      (blank spaces). Traversal cannot pass through non-zero cells of a different color.
      The consecutive-blank counter resets whenever a same-color cell is reached.
    - Coordinates are 0-indexed and inclusive.
    - Lists of boxes are sorted lexicographically for determinism.
    - Returns an empty dict if the grid is empty or contains no non-zero cells.
    """
    if not grid or not grid[0]:
        return {}

    rows, cols = len(grid), len(grid[0])
    # Clamp negative values to 0 to preserve expected behavior
    blank_hops = max(0, int(blank_hops))

    # Tracks which cells of a given color have already been assigned to any component
    visited_color = [[False] * cols for _ in range(rows)]
    boxes_by_color: Dict[int, List[Tuple[int, int, int, int]]] = {}

    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color == 0 or visited_color[r][c]:
                continue

            # BFS for the current component, allowing up to `blank_hops` consecutive blanks (0s)
            queue = deque()
            queue.append((r, c, 0))  # (row, col, consecutive_blank_count)
            visited_color[r][c] = True
            top = bottom = r
            left = right = c

            # Track best (lowest) consecutive-blank count we've seen for zero cells in this BFS
            visited_zero: List[List[int | None]] = [[None] * cols for _ in range(rows)]

            while queue:
                cr, cc, consec = queue.popleft()

                # Update bounding box whenever we visit a same-color cell
                if grid[cr][cc] == color:
                    if cr < top:
                        top = cr
                    if cr > bottom:
                        bottom = cr
                    if cc < left:
                        left = cc
                    if cc > right:
                        right = cc

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue

                    cell = grid[nr][nc]
                    if cell == color:
                        if not visited_color[nr][nc]:
                            visited_color[nr][nc] = True
                            queue.append((nr, nc, 0))  # reset consecutive blanks on reaching color
                    elif cell == 0:
                        new_consec = consec + 1
                        if new_consec <= blank_hops:
                            vz = visited_zero[nr][nc]
                            if vz is None or new_consec < vz:
                                visited_zero[nr][nc] = new_consec
                                queue.append((nr, nc, new_consec))
                    else:
                        # Different non-zero color: cannot traverse
                        continue

            boxes_by_color.setdefault(color, []).append((top, left, bottom, right))

    # Sort each color's boxes for determinism
    for color in boxes_by_color:
        boxes_by_color[color].sort()

    return boxes_by_color

def describe_grid(grid: List[List[int]], name: str = "grid", blank_hops: int = 0) -> str:
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
    
    # Append per-color bounding box information
    bboxes_by_color = get_largest_bounding_box(grid, blank_hops=blank_hops)
    if bboxes_by_color:
        description.append("- Bounding boxes by color (top, left, bottom, right):")
        for color in sorted(bboxes_by_color.keys()):
            boxes = bboxes_by_color[color]
            description.append(f"  - {color}: {len(boxes)} component(s)")
            for (top, left, bottom, right) in boxes:
                height = bottom - top + 1
                width = right - left + 1
                description.append(f"    - ({top}, {left}, {bottom}, {right}) size={height}x{width}")
    
    return "\n".join(description)

def build_scene_description(input_grid: List[List[int]], output_grid: List[List[int]]) -> str:
    """
    Build a scene description comparing input and output grids.
    """
    input_desc = describe_grid(input_grid, "input")
    output_desc = describe_grid(output_grid, "output")
    
    return f"SCENE DESCRIPTION\n\n{input_desc}\n\n{output_desc}"


def compare_grids(expected_grid: List[List[int]], actual_grid: List[List[int]]) -> Tuple[float, str]:
    """
    Compare two grids and return a description of the differences.

    Returns:
        Tuple[float, str]: A tuple containing the comparison score and a description of the differences.
    """
    if expected_grid == actual_grid:
        return 1.0, "The grids are identical."

    # If grids are different sizes, return 0.1
    if len(expected_grid) != len(actual_grid) or len(expected_grid[0]) != len(actual_grid[0]):
        return 0.1, "The grids are different sizes."

    message = "The grid sizes are the same."
    
    # Build a set of colors and counts
    
    # Convert the color counts to language
    expected_color_counts = dict(sorted(get_color_counts(expected_grid).items()))
    actual_color_counts = dict(sorted(get_color_counts(actual_grid).items()))

    
    if expected_color_counts == actual_color_counts:
        message += " The color counts match in both grids."
    else:
        for color, count in expected_color_counts.items():
            if color not in actual_color_counts:
                message += f"\nColor {color} is missing from the actual grid."
            elif actual_color_counts[color] != count:
                message += f"\nColor {color} has a different count in the actual grid."
            else:
                message += f"\nColor {color} has the correct count in the actual grid."
        
        for color, count in actual_color_counts.items():
            if color not in expected_color_counts:
                message += f"\nColor {color} is not in expected grid."

    # Compute a grid comparison score based on the number of matching cells
    matching_cells = 0
    total_cells = 0
    differing_cells = []
    for i in range(len(expected_grid)):
        for j in range(len(expected_grid[i])):
            total_cells += 1
            if expected_grid[i][j] == actual_grid[i][j]:
                matching_cells += 1
            else:
                differing_cells.append((i, j, expected_grid[i][j], actual_grid[i][j]))
    
    score = matching_cells / total_cells
    
    if differing_cells:
        message += f"\nThe following cells are different:\n"
        for row, col, expected, actual in differing_cells:
            message += f"Row {row}, Col {col}: Expected {expected}, Actual {actual}\n"

    return score, message