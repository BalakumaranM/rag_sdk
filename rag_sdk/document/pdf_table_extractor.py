"""Table grid detection and text extraction from PDF vector lines.

Uses intersection-based grid detection — no image processing or OpenCV needed.
"""

import logging
from typing import List, Optional, Tuple

from .pdf_models import BBox, GridCell, LineSegment, Table, TextSpan

logger = logging.getLogger(__name__)


class TableGrid:
    """Intermediate representation of a detected grid structure."""

    def __init__(
        self, x_coords: List[float], y_coords: List[float], bbox: BBox
    ) -> None:
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.bbox = bbox
        self.num_rows = len(y_coords) - 1
        self.num_cols = len(x_coords) - 1

    def get_cell_bbox(self, row: int, col: int) -> BBox:
        """Get the bounding box for a specific cell."""
        return BBox(
            x0=self.x_coords[col],
            y0=self.y_coords[row],
            x1=self.x_coords[col + 1],
            y1=self.y_coords[row + 1],
        )


class TableGridDetector:
    """Finds rectangular grids from horizontal and vertical line segments.

    Algorithm:
    1. Find intersection points of H and V segments
    2. Snap nearby intersections within tolerance
    3. Build grid from sorted unique X/Y coordinates
    4. Validate minimum rows and columns
    """

    def __init__(
        self,
        grid_snap_tolerance: float = 3.0,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
    ) -> None:
        self.grid_snap_tolerance = grid_snap_tolerance
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols

    def detect_grids(
        self,
        horizontal: List[LineSegment],
        vertical: List[LineSegment],
    ) -> List[TableGrid]:
        """Detect table grids from line segments.

        Args:
            horizontal: Horizontal line segments.
            vertical: Vertical line segments.

        Returns:
            List of detected TableGrid objects.
        """
        if not horizontal or not vertical:
            return []

        intersections = self._find_intersections(horizontal, vertical)
        if len(intersections) < 4:
            return []

        intersections = self._snap_points(intersections)

        x_vals = sorted(set(x for x, _ in intersections))
        y_vals = sorted(set(y for _, y in intersections))

        if len(x_vals) < 2 or len(y_vals) < 2:
            return []

        grids = self._build_grids(x_vals, y_vals, intersections)
        return [
            g
            for g in grids
            if g.num_rows >= self.min_table_rows and g.num_cols >= self.min_table_cols
        ]

    def _find_intersections(
        self,
        horizontal: List[LineSegment],
        vertical: List[LineSegment],
    ) -> List[Tuple[float, float]]:
        """Find intersection points between horizontal and vertical segments."""
        points: List[Tuple[float, float]] = []

        for h in horizontal:
            h_y = (h.y0 + h.y1) / 2
            h_x_min = min(h.x0, h.x1)
            h_x_max = max(h.x0, h.x1)

            for v in vertical:
                v_x = (v.x0 + v.x1) / 2
                v_y_min = min(v.y0, v.y1)
                v_y_max = max(v.y0, v.y1)

                if h_x_min <= v_x <= h_x_max and v_y_min <= h_y <= v_y_max:
                    points.append((v_x, h_y))

        return points

    def _snap_points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Snap nearby points to the same coordinates."""
        if not points:
            return []

        snapped_x = self._snap_values([p[0] for p in points])
        snapped_y = self._snap_values([p[1] for p in points])

        return list(set(zip(snapped_x, snapped_y)))

    def _snap_values(self, values: List[float]) -> List[float]:
        """Snap nearby values to representative values."""
        if not values:
            return []

        sorted_vals = sorted(set(values))
        groups: List[List[float]] = [[sorted_vals[0]]]

        for val in sorted_vals[1:]:
            if val - groups[-1][-1] <= self.grid_snap_tolerance:
                groups[-1].append(val)
            else:
                groups.append([val])

        snap_map: dict[float, float] = {}
        for group in groups:
            representative = sum(group) / len(group)
            for val in group:
                snap_map[val] = representative

        return [snap_map.get(v, v) for v in values]

    def _build_grids(
        self,
        x_vals: List[float],
        y_vals: List[float],
        intersections: List[Tuple[float, float]],
    ) -> List[TableGrid]:
        """Build grid structures from coordinate lists.

        Finds the largest contiguous rectangular region where all corner
        intersections exist.
        """
        intersection_set = set(intersections)

        # Find contiguous X and Y ranges where corners exist
        valid_x = self._find_contiguous_coords(x_vals, y_vals, intersection_set, "x")
        valid_y = self._find_contiguous_coords(y_vals, valid_x, intersection_set, "y")

        if len(valid_x) < 2 or len(valid_y) < 2:
            return []

        bbox = BBox(
            x0=valid_x[0],
            y0=valid_y[0],
            x1=valid_x[-1],
            y1=valid_y[-1],
        )

        return [TableGrid(x_coords=valid_x, y_coords=valid_y, bbox=bbox)]

    def _find_contiguous_coords(
        self,
        primary_coords: List[float],
        secondary_coords: List[float],
        intersection_set: set[Tuple[float, float]],
        axis: str,
    ) -> List[float]:
        """Find coordinates that form a contiguous grid with sufficient intersections."""
        valid: List[float] = []
        for coord in primary_coords:
            hits = 0
            for sec in secondary_coords:
                point = (coord, sec) if axis == "x" else (sec, coord)
                if point in intersection_set:
                    hits += 1
            # Need at least 2 intersections to be part of a grid
            if hits >= 2:
                valid.append(coord)
        return valid


class TableExtractor:
    """Maps text spans into detected table grids to produce structured Tables."""

    def extract_tables(
        self,
        grids: List[TableGrid],
        spans: List[TextSpan],
        detect_headers: bool = True,
    ) -> List[Table]:
        """Extract structured tables from grids and text spans.

        Args:
            grids: Detected table grids.
            spans: All text spans on the page.
            detect_headers: Whether to detect header rows via bold font heuristic.

        Returns:
            List of Table objects with headers and rows.
        """
        tables: List[Table] = []

        for grid in grids:
            cells = self._map_spans_to_cells(grid, spans)
            table = self._cells_to_table(grid, cells, detect_headers)
            if table:
                tables.append(table)

        return tables

    def _map_spans_to_cells(
        self, grid: TableGrid, spans: List[TextSpan]
    ) -> List[GridCell]:
        """Map text spans to grid cells based on span center point."""
        cells: List[GridCell] = []

        for row in range(grid.num_rows):
            for col in range(grid.num_cols):
                cell_bbox = grid.get_cell_bbox(row, col)
                cell_text_parts: List[str] = []

                for span in spans:
                    center_x = span.bbox.mid_x
                    center_y = span.bbox.mid_y
                    if cell_bbox.contains_point(center_x, center_y):
                        cell_text_parts.append(span.text)

                cells.append(
                    GridCell(
                        row=row,
                        col=col,
                        bbox=cell_bbox,
                        text=" ".join(cell_text_parts),
                    )
                )

        return cells

    def _cells_to_table(
        self,
        grid: TableGrid,
        cells: List[GridCell],
        detect_headers: bool,
    ) -> Optional[Table]:
        """Convert grid cells into a structured Table."""
        if not cells:
            return None

        # Organize cells into rows
        row_data: List[List[str]] = []
        for row_idx in range(grid.num_rows):
            row_cells = sorted(
                [c for c in cells if c.row == row_idx], key=lambda c: c.col
            )
            row_data.append([c.text for c in row_cells])

        if not row_data:
            return None

        if detect_headers and len(row_data) > 1:
            headers = row_data[0]
            rows = row_data[1:]
        else:
            headers = [f"col_{i}" for i in range(grid.num_cols)]
            rows = row_data

        return Table(
            headers=headers,
            rows=rows,
            bbox=grid.bbox,
            confidence=1.0,
        )
