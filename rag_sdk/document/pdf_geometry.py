"""Pure geometry utilities for PDF parsing.

No fitz/PyMuPDF dependency — all functions operate on data models.
"""

from typing import List, Tuple

from .pdf_models import BBox, LineSegment, PageElement, TextLine, TextSpan


def cluster_by_y(spans: List[TextSpan], tolerance: float = 2.0) -> List[TextLine]:
    """Group TextSpans into TextLines by Y-proximity.

    Sorts spans by mid_y, walks through and splits into clusters whenever
    the gap between consecutive mid_y values exceeds tolerance.
    Each cluster becomes a TextLine with spans sorted left-to-right.

    Args:
        spans: TextSpans extracted from a page.
        tolerance: Maximum vertical distance between spans in the same line.

    Returns:
        List of TextLines, sorted top-to-bottom.
    """
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: s.bbox.mid_y)
    clusters: List[List[TextSpan]] = [[sorted_spans[0]]]

    for span in sorted_spans[1:]:
        prev_mid_y = clusters[-1][-1].bbox.mid_y
        if abs(span.bbox.mid_y - prev_mid_y) <= tolerance:
            clusters[-1].append(span)
        else:
            clusters.append([span])

    lines: List[TextLine] = []
    for cluster in clusters:
        cluster.sort(key=lambda s: s.bbox.x0)
        bbox = BBox(
            x0=min(s.bbox.x0 for s in cluster),
            y0=min(s.bbox.y0 for s in cluster),
            x1=max(s.bbox.x1 for s in cluster),
            y1=max(s.bbox.y1 for s in cluster),
        )
        lines.append(TextLine(spans=cluster, bbox=bbox))

    return lines


def merge_touching_segments(
    segments: List[LineSegment], gap_tolerance: float = 2.0
) -> List[LineSegment]:
    """Merge collinear segments with overlapping or adjacent ranges.

    For horizontal segments: groups by similar Y, merges overlapping X ranges.
    For vertical segments: groups by similar X, merges overlapping Y ranges.

    Args:
        segments: Line segments to merge (should be all horizontal or all vertical).
        gap_tolerance: Maximum gap between segments to consider them adjacent.

    Returns:
        Merged list of segments.
    """
    if not segments:
        return []

    # Determine orientation from first segment
    is_horizontal = segments[0].is_horizontal

    if is_horizontal:
        # Group by Y, merge X ranges
        return _merge_segments_by_axis(
            segments,
            group_key=lambda s: round(s.y0, 1),
            range_start=lambda s: min(s.x0, s.x1),
            range_end=lambda s: max(s.x0, s.x1),
            make_segment=lambda y, x0, x1: LineSegment(x0=x0, y0=y, x1=x1, y1=y),
            gap_tolerance=gap_tolerance,
        )
    else:
        # Group by X, merge Y ranges
        return _merge_segments_by_axis(
            segments,
            group_key=lambda s: round(s.x0, 1),
            range_start=lambda s: min(s.y0, s.y1),
            range_end=lambda s: max(s.y0, s.y1),
            make_segment=lambda x, y0, y1: LineSegment(x0=x, y0=y0, x1=x, y1=y1),
            gap_tolerance=gap_tolerance,
        )


def _merge_segments_by_axis(
    segments: List[LineSegment],
    group_key,  # type: ignore[type-arg]
    range_start,  # type: ignore[type-arg]
    range_end,  # type: ignore[type-arg]
    make_segment,  # type: ignore[type-arg]
    gap_tolerance: float,
) -> List[LineSegment]:
    """Helper: group segments by one axis, merge ranges on the other."""
    groups: dict[float, List[LineSegment]] = {}
    for seg in segments:
        key = group_key(seg)
        matched = False
        for existing_key in groups:
            if abs(existing_key - key) <= gap_tolerance:
                groups[existing_key].append(seg)
                matched = True
                break
        if not matched:
            groups[key] = [seg]

    merged: List[LineSegment] = []
    for axis_val, group in groups.items():
        ranges = sorted(
            [(range_start(s), range_end(s)) for s in group], key=lambda r: r[0]
        )
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= current_end + gap_tolerance:
                current_end = max(current_end, end)
            else:
                merged.append(make_segment(axis_val, current_start, current_end))
                current_start, current_end = start, end
        merged.append(make_segment(axis_val, current_start, current_end))

    return merged


def classify_segments(
    segments: List[LineSegment], min_length: float = 10.0
) -> Tuple[List[LineSegment], List[LineSegment]]:
    """Split line segments into horizontal and vertical lists, discarding diagonals.

    Args:
        segments: All line segments from the page.
        min_length: Minimum segment length to keep.

    Returns:
        Tuple of (horizontal_segments, vertical_segments).
    """
    horizontal: List[LineSegment] = []
    vertical: List[LineSegment] = []

    for seg in segments:
        if seg.length < min_length:
            continue
        if seg.is_horizontal:
            horizontal.append(seg)
        elif seg.is_vertical:
            vertical.append(seg)

    return horizontal, vertical


def sort_reading_order(
    elements: List[PageElement], y_tolerance: float = 5.0
) -> List[PageElement]:
    """Sort page elements in reading order: top-to-bottom, left-to-right.

    Groups elements into rows by Y-proximity, then sorts each row left-to-right.

    Args:
        elements: Page elements to sort.
        y_tolerance: Maximum Y distance to consider elements on the same row.

    Returns:
        Elements in reading order.
    """
    if not elements:
        return []

    sorted_elements = sorted(elements, key=lambda e: e.bbox.mid_y)
    rows: List[List[PageElement]] = [[sorted_elements[0]]]

    for elem in sorted_elements[1:]:
        prev_mid_y = rows[-1][0].bbox.mid_y
        if abs(elem.bbox.mid_y - prev_mid_y) <= y_tolerance:
            rows[-1].append(elem)
        else:
            rows.append([elem])

    result: List[PageElement] = []
    for row in rows:
        row.sort(key=lambda e: e.bbox.x0)
        result.extend(row)

    return result
