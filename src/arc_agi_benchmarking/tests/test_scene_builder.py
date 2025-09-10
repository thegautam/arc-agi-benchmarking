import pytest

# We import the module so the test file loads even if the function isn't implemented yet.
from arc_agi_benchmarking.prompts import scene_builder


def test_get_largest_bounding_box_simple():
    """
    Verifies that get_largest_bounding_box returns a dict mapping each color
    to a list of bounding boxes for its 4-connected components.

    Expected return format for each bbox: (top, left, bottom, right), 0-indexed, inclusive.
    Zeros are treated as background and ignored. We check for both color 1 and 2.
    """
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 2, 2],
        [0, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2, 2],
    ]

    expected_bbox_color_1 = (1, 1, 2, 2)
    expected_bbox_color_2 = (2, 4, 4, 5)

    result = scene_builder.get_largest_bounding_box(grid)

    # Ensure structure: dict[color] -> list of bboxes
    assert isinstance(result, dict)
    assert 1 in result and 2 in result
    assert isinstance(result[1], list) and isinstance(result[2], list)

    # Expect one component for color 1 and one for color 2 in this grid
    assert len(result[1]) == 1
    assert len(result[2]) == 1

    # Check exact bounding boxes
    assert result[1][0] == expected_bbox_color_1
    assert result[2][0] == expected_bbox_color_2


def test_multiple_components_same_color():
    """
    A grid where color 2 forms two separate components. Verify both bboxes are
    present and sorted lexicographically.
    """
    grid = [
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
    ]

    expected_boxes_color_2 = [
        (0, 1, 1, 2),  # top 2x2 block
        (3, 0, 4, 0),  # vertical 2x1 on the left
    ]

    result = scene_builder.get_largest_bounding_box(grid)

    assert isinstance(result, dict)
    # Only color 2 should be present
    assert set(result.keys()) == {2}
    assert isinstance(result[2], list)
    assert len(result[2]) == 2
    # Lists are sorted lexicographically
    assert result[2] == expected_boxes_color_2


def test_odd_sized_components_and_diagonal_non_connection():
    """
    A grid with odd-sized components and diagonal cells of the same color that
    are not 4-connected (should be separate components).
    """
    grid = [
        [1, 0, 0, 0, 3],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 3, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
    ]

    # Color 1: vertical 3x1 at left, and horizontal 1x3 at bottom row
    expected_boxes_color_1 = [
        (0, 0, 2, 0),  # vertical stripe (odd height 3)
        (4, 1, 4, 3),  # horizontal stripe (odd width 3)
    ]

    # Color 3: two isolated single cells, diagonally separated -> two components
    expected_boxes_color_3 = [
        (0, 4, 0, 4),
        (2, 3, 2, 3),
    ]

    result = scene_builder.get_largest_bounding_box(grid)

    assert isinstance(result, dict)
    assert 1 in result and 3 in result

    # Color 1 checks
    assert isinstance(result[1], list)
    assert len(result[1]) == 2
    assert result[1] == expected_boxes_color_1

    # Color 3 checks
    assert isinstance(result[3], list)
    assert len(result[3]) == 2
    assert result[3] == expected_boxes_color_3
