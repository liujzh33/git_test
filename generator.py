from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List

import geopandas as gpd
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import polygonize, unary_union, voronoi_diagram
from shapely.prepared import prep


@dataclass
class GenerationConfig:
    x_min: int = 0
    y_min: int = 0
    x_max: int = 500
    y_max: int = 500
    red_count_min: int = 6
    red_count_max: int = 10
    cell_size: int = 10
    min_shared_edge_m: float = 2.0
    green_area_ratio_min: float = 0.05
    green_area_ratio_max: float = 0.25
    green_zone_count_min: int = 3
    green_zone_count_max: int = 3
    roughness_strength: float = 0.0
    seed: int = 42
    minerals_per_red_min: int = 1
    minerals_per_red_max: int = 3


def _sample_points_in_bbox(bbox: Polygon, n: int, rng: random.Random) -> List[Point]:
    minx, miny, maxx, maxy = bbox.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    min_dist = diag / (n * 0.80)
    pts: List[Point] = []
    tries = 0
    while len(pts) < n and tries < 25000:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if pts and min(p.distance(pp) for pp in pts) < min_dist:
            continue
        pts.append(p)
    return pts


def _sample_points_in_polygon(poly: Polygon, n: int, rng: random.Random, min_dist: float) -> List[Point]:
    minx, miny, maxx, maxy = poly.bounds
    pts: List[Point] = []
    tries = 0
    while len(pts) < n and tries < 25000:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if not p.within(poly):
            continue
        if pts and min(p.distance(pp) for pp in pts) < min_dist:
            continue
        pts.append(p)
    return pts


def _shared_boundary_length(a: Polygon, b: Polygon) -> float:
    return a.boundary.intersection(b.boundary).length


def _build_red_polygons(bbox: Polygon, red_count: int, cfg: GenerationConfig, rng: random.Random):
    bbox_area = float(bbox.area)
    # 小区域（如 100x100）下，min_red_area 不能太大，否则会导致生成失败或区域缺失
    min_red_area = max(5.0, bbox_area * 0.005)

    seeds = _sample_points_in_bbox(bbox, red_count, rng)
    if len(seeds) < red_count:
        return []
    mp = MultiPoint(seeds)
    vor = voronoi_diagram(mp, envelope=bbox.envelope, edges=False)
    cells = [g for g in vor.geoms if isinstance(g, Polygon)]
    # 直接用 Voronoi 单元 clip 到 bbox：Voronoi 单元是一个天然分区，但浮点裁剪仍可能产生微小重叠/洞
    raw_result: List[Polygon] = []
    for c in cells:
        clipped = c.intersection(bbox).buffer(0)
        if clipped.is_empty or clipped.area < min_red_area:
            continue
        # clip 后可能产生 MultiPolygon：取最大连通分量，避免渲染时“外环填充导致的视觉重叠”
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.is_empty or clipped.area < min_red_area:
            continue
        raw_result.append(clipped)

    if len(raw_result) != red_count:
        return []

    # 强制无重叠重分区：把所有红区边界线 polygonize 成不重叠的面片，
    # 再根据面片质心归属到原始 red cell，最后合并回每个 red cell。
    # 这样保证：每个红区之间不真正覆盖，从而避免 Bokeh 仅画外环导致的“视觉重叠”。
    prepared_raw = [prep(p) for p in raw_result]
    all_linework = unary_union([p.boundary for p in raw_result])
    pieces = list(polygonize(all_linework))

    # 归属面片到原始 red index
    bucket: List[List[Polygon]] = [[] for _ in range(red_count)]
    for piece in pieces:
        if piece.is_empty or piece.area <= 1e-8:
            continue
        # 面片可能很细碎，按面积过滤掉明显噪声
        if piece.area < min_red_area * 0.05:
            continue
        rp = piece.representative_point()
        idx = None
        for i, pr in enumerate(prepared_raw):
            if pr.contains(rp):
                idx = i
                break
        if idx is None:
            continue
        bucket[idx].append(piece)

    # 合并面片回每个 red cell
    result: List[Polygon] = []
    for i in range(red_count):
        if not bucket[i]:
            # 失败则回退到原始 raw_result（至少不会直接报错）
            result = raw_result
            break
        merged = unary_union(bucket[i]).buffer(0)
        # merged 可能是 MultiPolygon，取最大连通分量
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        if merged.is_empty or merged.area < min_red_area:
            result = raw_result
            break
        result.append(merged)

    if len(result) != red_count:
        result = raw_result

    # Shared edge hard check（小区域放宽）
    min_edge = max(1.0, cfg.min_shared_edge_m)
    for i, poly in enumerate(result):
        max_shared = 0.0
        for j, other in enumerate(result):
            if i == j:
                continue
            max_shared = max(max_shared, _shared_boundary_length(poly, other))
        if max_shared < min_edge:
            return []
    return result


def _regions_to_gdf(regions: List[Polygon], cfg: GenerationConfig, rng: random.Random) -> gpd.GeoDataFrame:
    mineral_keys = ["FeO", "TiO", "Al2O3", "SiO2"]
    geoms = []
    attrs = []
    for i, poly in enumerate(regions, start=1):
        geoms.append(poly.buffer(0))
        picked_n = rng.randint(cfg.minerals_per_red_min, cfg.minerals_per_red_max)
        picked = rng.sample(mineral_keys, picked_n)
        values = {k: 0.0 for k in mineral_keys}
        for k in picked:
            if k == "FeO":
                values[k] = round(rng.uniform(22, 55), 2)
            elif k == "TiO":
                values[k] = round(rng.uniform(3, 14), 2)
            elif k == "Al2O3":
                values[k] = round(rng.uniform(8, 28), 2)
            elif k == "SiO2":
                values[k] = round(rng.uniform(10, 50), 2)
        attrs.append(
            {
                "red_id": f"R{i:02d}",
                **values,
                "minerals": ",".join(picked),
                "color_value": round(sum(values.values()), 3),
            }
        )
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs=None)


def _make_irregular_polygon(center: Point, base_r: float, rng: random.Random, n_vertices: int = 7) -> Polygon:
    pts = []
    for i in range(n_vertices):
        ang = (2.0 * 3.1415926 * i / n_vertices) + rng.uniform(-0.22, 0.22)
        rr = base_r * rng.uniform(0.70, 1.30)
        pts.append((center.x + rr * math.cos(ang), center.y + rr * math.sin(ang)))
    return Polygon(pts).buffer(0)


def _make_green_zones(red_union: Polygon, cfg: GenerationConfig, rng: random.Random) -> gpd.GeoDataFrame:
    diag = (red_union.bounds[2] - red_union.bounds[0]) ** 2 + (red_union.bounds[3] - red_union.bounds[1]) ** 2
    diag = max(1e-6, diag) ** 0.5
    inset = min(diag * 0.13, 70.0)

    safe = red_union.buffer(-inset).buffer(0)
    if safe.is_empty:
        safe = red_union.buffer(-inset * 0.5).buffer(0)
    if safe.is_empty:
        safe = red_union

    zone_count = rng.randint(cfg.green_zone_count_min, cfg.green_zone_count_max)
    total_target = red_union.area * rng.uniform(cfg.green_area_ratio_min, cfg.green_area_ratio_max)
    weights = [rng.uniform(0.85, 1.15) for _ in range(zone_count)]
    s = sum(weights)
    targets = [total_target * w / s for w in weights]

    minx, miny, maxx, maxy = safe.bounds
    centers: List[Point] = []
    zones: List[dict] = []
    min_center_dist = max(diag * 0.22, cfg.cell_size * 0.6)

    for i in range(zone_count):
        center = None
        for _ in range(5000):
            p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if not p.within(safe):
                continue
            if centers and min(p.distance(c) for c in centers) < min_center_dist:
                continue
            center = p
            break
        if center is None:
            # Fallback with relaxed spacing.
            for _ in range(3000):
                p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if p.within(safe):
                    center = p
                    break
        if center is None:
            continue
        centers.append(center)

        base_r = (targets[i] / 3.1415926) ** 0.5
        zone = _make_irregular_polygon(center, base_r * 0.9, rng, n_vertices=rng.randint(6, 9)).intersection(safe).buffer(0)
        # Keep zones separated (no touching / overlap) by removing overlap and enforcing spacing.
        if zones:
            existing_union = unary_union([z["geometry"] for z in zones]).buffer(0)
            separation = max(diag * 0.06, cfg.cell_size * 0.25)
            zone = zone.difference(existing_union.buffer(separation)).buffer(0)
        min_zone_area = max(15.0, red_union.area * 0.008)
        if zone.is_empty or zone.area < min_zone_area:
            continue
        if zone.geom_type == "MultiPolygon":
            zone = max(zone.geoms, key=lambda g: g.area)
        zones.append({"green_id": f"G{i+1:02d}", "geometry": zone})

    if len(zones) < zone_count:
        return gpd.GeoDataFrame([], geometry=[], crs=None)
    return gpd.GeoDataFrame(zones[:zone_count], geometry="geometry", crs=None)


def generate_red_and_green(cfg: GenerationConfig):
    rng = random.Random(cfg.seed)
    bbox = box(cfg.x_min, cfg.y_min, cfg.x_max, cfg.y_max)
    for _ in range(120):
        red_count = rng.randint(cfg.red_count_min, cfg.red_count_max)
        regions = _build_red_polygons(bbox, red_count, cfg, rng)
        if not regions:
            continue

        red_gdf = _regions_to_gdf(regions, cfg, rng)
        red_union = unary_union(red_gdf.geometry.tolist()).buffer(0).intersection(bbox)
        if red_union.is_empty or abs(red_union.area - bbox.area) > 1e-5:
            continue

        green_gdf = _make_green_zones(red_union, cfg, rng)
        if green_gdf.empty:
            continue
        green_union = unary_union(green_gdf.geometry.tolist()).buffer(0)
        ratio = green_union.area / red_union.area
        if not (cfg.green_area_ratio_min <= ratio <= cfg.green_area_ratio_max):
            continue
        if not green_union.within(red_union.buffer(1e-7)):
            continue
        return red_gdf, green_gdf, bbox

    raise RuntimeError("Failed to generate valid constrained geometry. Try changing seed or parameters.")
