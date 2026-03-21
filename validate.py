from __future__ import annotations

from dataclasses import dataclass
from typing import List

import geopandas as gpd
from shapely.ops import unary_union


@dataclass
class ValidationReport:
    red_count: int
    overlap_pairs: int
    green_count: int
    green_within_red_union: bool
    black_cells: int
    clipped_ratio: float
    unassigned_cells: int
    min_max_shared_edge: float


def _shared_boundary_length(a, b) -> float:
    return a.boundary.intersection(b.boundary).length


def validate_all(red_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame, black_gdf: gpd.GeoDataFrame) -> ValidationReport:
    overlap_pairs = 0
    per_region_max_shared: List[float] = []
    for i in range(len(red_gdf)):
        max_shared_for_i = 0.0
        for j in range(i + 1, len(red_gdf)):
            ai = red_gdf.geometry.iloc[i]
            bj = red_gdf.geometry.iloc[j]
            inter_area = ai.intersection(bj).area
            if inter_area > 1e-6:
                overlap_pairs += 1
            shared = _shared_boundary_length(ai, bj)
            max_shared_for_i = max(max_shared_for_i, shared)
        # check neighbors before i as well
        for j in range(i):
            ai = red_gdf.geometry.iloc[i]
            bj = red_gdf.geometry.iloc[j]
            shared = _shared_boundary_length(ai, bj)
            max_shared_for_i = max(max_shared_for_i, shared)
        per_region_max_shared.append(max_shared_for_i)

    red_union = unary_union(red_gdf.geometry.tolist()).buffer(0)
    green_union = unary_union(green_gdf.geometry.tolist()).buffer(0)
    green_within_red_union = green_union.buffer(-1e-7).within(red_union.buffer(1e-7))

    black_cells = len(black_gdf)
    clipped_ratio = float(black_gdf["is_clipped"].mean()) if black_cells else 0.0
    unassigned_cells = int(black_gdf["red_id"].isna().sum())
    min_max_shared_edge = min(per_region_max_shared) if per_region_max_shared else 0.0

    return ValidationReport(
        red_count=len(red_gdf),
        overlap_pairs=overlap_pairs,
        green_count=len(green_gdf),
        green_within_red_union=green_within_red_union,
        black_cells=black_cells,
        clipped_ratio=clipped_ratio,
        unassigned_cells=unassigned_cells,
        min_max_shared_edge=min_max_shared_edge,
    )
