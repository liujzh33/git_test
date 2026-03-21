from __future__ import annotations

import math
from typing import List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep


def build_clipped_grid(green_geom, bbox_geom, cell_size: int = 100) -> gpd.GeoDataFrame:
    """生成规则正方形网格，只保留质心在绿色区域内的格子（完整正方形，不裁剪）。"""
    del bbox_geom  # kept for API compatibility
    target_area = float(cell_size * cell_size)
    records: List[dict] = []
    cell_id = 1

    green_parts = [green_geom] if isinstance(green_geom, Polygon) else list(green_geom.geoms)
    for part in green_parts:
        part = part.buffer(0)
        if part.is_empty or part.area <= 0:
            continue
        prep_part = prep(part)
        minx, miny, maxx, maxy = part.bounds

        x0 = int(minx // cell_size) * cell_size
        y0 = int(miny // cell_size) * cell_size
        x1 = int((maxx + cell_size - 1) // cell_size) * cell_size
        y1 = int((maxy + cell_size - 1) // cell_size) * cell_size

        half = cell_size * 0.5
        for cx in range(x0, x1, cell_size):
            for cy in range(y0, y1, cell_size):
                centroid = Point(cx + half, cy + half)
                if not prep_part.contains(centroid):
                    continue
                sq = box(cx, cy, cx + cell_size, cy + cell_size)
                records.append(
                    {
                        "cell_id": f"C{cell_id:05d}",
                        "geometry": sq,
                        "is_clipped": False,
                        "cell_area": target_area,
                    }
                )
                cell_id += 1

    return gpd.GeoDataFrame(records, geometry="geometry", crs=None)


def assign_properties_to_cells(cells_gdf: gpd.GeoDataFrame, red_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = cells_gdf.copy()
    work["centroid"] = work.geometry.centroid
    centroids = gpd.GeoDataFrame(work[["cell_id"]], geometry=work["centroid"], crs=None)

    joined = gpd.sjoin(
        centroids,
        red_gdf[["red_id", "FeO", "TiO", "Al2O3", "SiO2", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    missing = joined["red_id"].isna()
    if missing.any():
        fallback = gpd.sjoin_nearest(
            centroids.loc[missing],
            red_gdf[["red_id", "FeO", "TiO", "Al2O3", "SiO2", "geometry"]],
            how="left",
        ).drop(columns=["index_right"], errors="ignore")
        joined.loc[missing, ["red_id", "FeO", "TiO", "Al2O3", "SiO2"]] = fallback[
            ["red_id", "FeO", "TiO", "Al2O3", "SiO2"]
        ].to_numpy()

    out = work.merge(
        joined[["cell_id", "red_id", "FeO", "TiO", "Al2O3", "SiO2"]],
        on="cell_id",
        how="left",
    )
    out = out.drop(columns=["centroid"])

    ordered_cols = [
        "cell_id",
        "red_id",
        "FeO",
        "TiO",
        "Al2O3",
        "SiO2",
        "cell_area",
        "is_clipped",
        "geometry",
    ]
    return out[ordered_cols]


def export_outputs(result_gdf: gpd.GeoDataFrame, out_geojson: str, out_csv: str) -> None:
    result_gdf.to_file(out_geojson, driver="GeoJSON")
    # WKT keeps geometry available in plain CSV for office presentation.
    df = pd.DataFrame(result_gdf.drop(columns="geometry"))
    df["geometry_wkt"] = result_gdf.geometry.to_wkt()
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
