from __future__ import annotations

import os
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.io as pio
import geopandas as gpd
import plotly.graph_objects as go
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, HoverTool, Spinner, Tabs
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union, triangulate

from generator import GenerationConfig, generate_red_and_green
from grid_assign import build_clipped_grid, assign_properties_to_cells


MINERALS = ["FeO", "TiO", "Al2O3", "SiO2"]
MINERAL_COLORS = {
    "FeO": "#d73027",
    "TiO": "#4575b4",
    "Al2O3": "#1a9850",
    "SiO2": "#984ea3",
}


def _geom_to_patches(geom, simplify_tol: float) -> List[Tuple[List[float], List[float]]]:
    """Convert (Multi)Polygon to Bokeh patch rings (exterior only)."""
    if geom is None or geom.is_empty:
        return []

    g = geom
    try:
        g = g.simplify(simplify_tol, preserve_topology=True) if simplify_tol > 0 else g
    except Exception:
        pass

    if isinstance(g, Polygon):
        # 矩形快速路径：避免 exterior.xy 开销
        if len(g.exterior.coords) == 5:
            minx, miny, maxx, maxy = g.bounds
            return [([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny])]
        x, y = g.exterior.xy
        return [(list(x), list(y))]
    if isinstance(g, MultiPolygon):
        out: List[Tuple[List[float], List[float]]] = []
        for part in g.geoms:
            out.extend(_geom_to_patches(part, simplify_tol))
        return out
    return []


def _dominant_mineral_from_amounts(amounts: Dict[str, float]) -> str:
    best_m = "FeO"
    best_v = -1.0
    for m in MINERALS:
        v = float(amounts.get(m, 0.0) or 0.0)
        if v > best_v:
            best_v = v
            best_m = m
    return best_m


def _mix_with_white(hex_color: str, color_weight: float = 0.22) -> str:
    """Blend a color with white to mimic low-alpha pastel, but keep opaque rendering."""
    h = (hex_color or "#cccccc").strip().lstrip("#")
    if len(h) != 6:
        h = "cccccc"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    rw = int(color_weight * r + (1.0 - color_weight) * 255)
    gw = int(color_weight * g + (1.0 - color_weight) * 255)
    bw = int(color_weight * b + (1.0 - color_weight) * 255)
    return f"#{rw:02x}{gw:02x}{bw:02x}"


def _tri_area_weighted_mineral_shares(
    overlaps: List[Tuple[float, Dict[str, float]]],
    cell_area: float,
) -> Dict[str, float]:
    """
    overlaps: list of (inter_area, red_amounts_by_mineral)
    return: mineral share percent (sum to 1)
    """
    amounts = {m: 0.0 for m in MINERALS}
    for inter_area, red_amounts in overlaps:
        if cell_area <= 0:
            continue
        prop = inter_area / cell_area
        for m in MINERALS:
            amounts[m] += prop * float(red_amounts.get(m, 0.0) or 0.0)
    total = sum(amounts.values())
    if total <= 0:
        return {m: 0.0 for m in MINERALS}
    return {m: amounts[m] / total for m in MINERALS}


def _total_mineral(row: Dict[str, float]) -> float:
    return sum(float(row.get(m, 0.0) or 0.0) for m in MINERALS)


def _triangles_from_polygon(poly: Polygon, simplify_tol: float) -> List[Polygon]:
    if poly is None or poly.is_empty or poly.area <= 1e-9:
        return []
    p = poly.simplify(simplify_tol, preserve_topology=True) if simplify_tol > 0 else poly
    if p.is_empty or p.area <= 1e-9:
        return []
    tris = triangulate(p)
    out: List[Polygon] = []
    iter_tris = getattr(tris, "geoms", tris) if tris is not None else []
    for t in iter_tris:
        if not isinstance(t, Polygon):
            continue
        if t.area <= 1e-9:
            continue
        if t.centroid.within(p.buffer(1e-9)):
            out.append(t)
    return out


def _triangles_from_geom(geom, simplify_tol: float) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return _triangles_from_polygon(geom, simplify_tol)
    if isinstance(geom, MultiPolygon):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_triangles_from_polygon(g, simplify_tol))
        return out
    return []


def _mesh_from_triangles(
    triangles: Iterable[Polygon],
    z_func,
    color: str,
) -> go.Mesh3d:
    xs, ys, zs, i, j, k = [], [], [], [], [], []
    for tri in triangles:
        coords = list(tri.exterior.coords)
        if len(coords) < 4:
            continue
        (x0, y0), (x1, y1), (x2, y2) = coords[0], coords[1], coords[2]
        z = float(z_func(tri))
        base_idx = len(xs)
        xs.extend([x0, x1, x2])
        ys.extend([y0, y1, y2])
        zs.extend([z, z, z])
        i.append(base_idx)
        j.append(base_idx + 1)
        k.append(base_idx + 2)
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color, opacity=0.95, flatshading=True)


def _build_3d_figure(
    red_data: dict,
    green_data: dict,
    black_data: dict,
    cfg: GenerationConfig,
) -> go.Figure:
    """连续山体：底色来自2D red分区，叠加2-4个高峰（最高1500m）。"""
    z_base = 20.0
    z_max = 1500.0

    def _patch_to_geom(px, py):
        if not px or not py or len(px) < 3 or len(py) < 3:
            return None
        try:
            p = Polygon(list(zip(px, py))).buffer(0)
            if p.is_empty or p.area <= 1e-9:
                return None
            return p
        except Exception:
            return None

    def _mix_with_white(hex_color: str, color_weight: float = 0.22) -> str:
        """模拟2D alpha=0.22 叠加白底后的浅色效果。"""
        h = (hex_color or "#cccccc").strip().lstrip("#")
        if len(h) != 6:
            h = "cccccc"
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        rw = int(color_weight * r + (1.0 - color_weight) * 255)
        gw = int(color_weight * g + (1.0 - color_weight) * 255)
        bw = int(color_weight * b + (1.0 - color_weight) * 255)
        return f"#{rw:02x}{gw:02x}{bw:02x}"

    def _smooth_closed_path(
        px: Iterable[float],
        py: Iterable[float],
        points_per_edge: int = 10,
        smooth_passes: int = 2,
    ) -> tuple[List[float], List[float]]:
        """Densify + smooth a closed polygon outline for continuous-looking 3D curves."""
        xs = np.asarray(list(px), dtype=float)
        ys = np.asarray(list(py), dtype=float)
        if xs.size < 3 or ys.size < 3 or xs.size != ys.size:
            return list(px), list(py)

        # Drop duplicated closing point, we'll close again at end.
        if abs(xs[0] - xs[-1]) < 1e-9 and abs(ys[0] - ys[-1]) < 1e-9:
            xs = xs[:-1]
            ys = ys[:-1]
        n = int(xs.size)
        if n < 3:
            return list(px), list(py)

        ppe = max(4, int(points_per_edge))
        dense_x: List[float] = []
        dense_y: List[float] = []
        for i in range(n):
            x0, y0 = float(xs[i]), float(ys[i])
            x1, y1 = float(xs[(i + 1) % n]), float(ys[(i + 1) % n])
            for t in np.linspace(0.0, 1.0, ppe, endpoint=False):
                dense_x.append((1.0 - t) * x0 + t * x1)
                dense_y.append((1.0 - t) * y0 + t * y1)

        sx = np.asarray(dense_x, dtype=float)
        sy = np.asarray(dense_y, dtype=float)
        for _ in range(max(0, int(smooth_passes))):
            sx = (np.roll(sx, 1) + 2.0 * sx + np.roll(sx, -1)) / 4.0
            sy = (np.roll(sy, 1) + 2.0 * sy + np.roll(sy, -1)) / 4.0

        out_x = sx.tolist()
        out_y = sy.tolist()
        out_x.append(out_x[0])
        out_y.append(out_y[0])
        return out_x, out_y

    # parse 2D red polygons/colors
    red_polys: List[Polygon] = []
    red_cols: List[str] = []
    for px, py, color in zip(red_data.get("xs", []), red_data.get("ys", []), red_data.get("fill_color", [])):
        g = _patch_to_geom(px, py)
        if g is None:
            continue
        red_polys.append(g)
        red_cols.append(_mix_with_white(color or "#cccccc", color_weight=0.22))
    if not red_polys:
        return go.Figure()

    # bounds/grid
    bds = [g.bounds for g in red_polys]
    minx = min(b[0] for b in bds)
    miny = min(b[1] for b in bds)
    maxx = max(b[2] for b in bds)
    maxy = max(b[3] for b in bds)
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5

    nx = 140
    ny = 140
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x, y)

    # choose 2-4 peak centers from large red polygons
    rng = random.Random(cfg.seed)
    n_peaks = int(rng.randint(2, 4))
    red_sorted = sorted(red_polys, key=lambda g: float(g.area), reverse=True)
    min_dist = max(300.0, diag * 0.18)
    peaks: List[tuple[float, float]] = []
    for poly in red_sorted:
        c = poly.centroid
        cand = (float(c.x), float(c.y))
        if peaks and min(((cand[0] - px) ** 2 + (cand[1] - py) ** 2) ** 0.5 for px, py in peaks) < min_dist:
            continue
        peaks.append(cand)
        if len(peaks) >= n_peaks:
            break
    if not peaks:
        c0 = red_sorted[0].centroid
        peaks = [(float(c0.x), float(c0.y))]
    peak_amps = [rng.uniform(0.85, 1.35) for _ in peaks]
    sigma = max(240.0, diag * 0.13)

    # continuous gaussian terrain field
    Zraw = np.zeros_like(X, dtype=float)
    for (px, py), amp in zip(peaks, peak_amps):
        d2 = (X - px) ** 2 + (Y - py) ** 2
        Zraw += amp * np.exp(-d2 / (2.0 * sigma * sigma))
    zmin = float(Zraw.min())
    zmax_raw = float(Zraw.max())
    zd = (zmax_raw - zmin) if (zmax_raw - zmin) != 0 else 1.0
    Z = z_base + (Zraw - zmin) / zd * (z_max - z_base)

    def _terrain_z(xv: float, yv: float) -> float:
        v = 0.0
        for (px, py), amp in zip(peaks, peak_amps):
            d2 = (xv - px) ** 2 + (yv - py) ** 2
            v += amp * math.exp(-d2 / (2.0 * sigma * sigma))
        return z_base + ((v - zmin) / zd) * (z_max - z_base)

    # classify each grid point by red polygon color (2D-consistent base colors)
    pts = [Point(float(xx), float(yy)) for yy in y for xx in x]
    pts_gdf = gpd.GeoDataFrame({"idx": np.arange(len(pts))}, geometry=pts, crs=None)
    red_gdf = gpd.GeoDataFrame({"fill_color": red_cols}, geometry=red_polys, crs=None)
    joined = gpd.sjoin(pts_gdf, red_gdf[["fill_color", "geometry"]], how="left", predicate="within").drop(
        columns=["index_right"], errors="ignore"
    )
    miss = joined["fill_color"].isna()
    if bool(miss.any()):
        near = gpd.sjoin_nearest(pts_gdf.loc[miss, :], red_gdf[["fill_color", "geometry"]], how="left").drop(
            columns=["index_right"], errors="ignore"
        )
        joined.loc[miss, ["fill_color"]] = near[["fill_color"]].to_numpy()
    color_flat = joined["fill_color"].fillna("#dddddd").astype(str).to_numpy()

    # discrete colorscale for many exact hex colors
    unique_colors = list(dict.fromkeys(color_flat.tolist()))
    cidx = {c: i for i, c in enumerate(unique_colors)}
    surfacecolor = np.array([float(cidx[c]) for c in color_flat], dtype=float).reshape(Y.shape)
    ncol = max(1, len(unique_colors))
    d = (ncol - 1) if ncol > 1 else 1.0
    colorscale = []
    for i, c in enumerate(unique_colors):
        p0 = (i / d) if d != 0 else 0.0
        p1 = 1.0 if i == (ncol - 1) else ((i + 1) / d)
        colorscale.append([p0, c])
        colorscale.append([p1, c])

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=Z,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            cmin=0,
            cmax=(ncol - 1),
            showscale=False,
            opacity=1.0,
            lighting=dict(
                ambient=0.98,
                diffuse=0.20,
                specular=0.06,
                roughness=1.0,
                fresnel=0.0,
            ),
            contours=dict(
                x=dict(show=True, color="rgba(120,120,120,0.28)", width=1),
                y=dict(show=True, color="rgba(120,120,120,0.28)", width=1),
                z=dict(show=False),
            ),
            # Let black-cell overlays own hover interaction in green zones.
            hoverinfo="skip",
        )
    )

    # drape green / black overlays on terrain
    for px, py in zip(green_data.get("xs", []), green_data.get("ys", [])):
        sx, sy = _smooth_closed_path(px, py, points_per_edge=12, smooth_passes=3)
        # Lift outlines above surface to avoid z-fighting flicker.
        zline = [_terrain_z(float(xi), float(yi)) + 16.0 for xi, yi in zip(sx, sy)]
        # White under-stroke improves readability on both light/purple/red areas.
        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=zline,
                mode="lines",
                line=dict(color="#ffffff", width=12),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=zline,
                mode="lines",
                line=dict(color="#00b050", width=7),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 黑色块：数量过多时只画山体+绿框，避免 Plotly 卡死
    black_xs = black_data.get("xs", [])
    black_ys = black_data.get("ys", [])
    black_colors = black_data.get("fill_color", [])
    black_tooltips = black_data.get("tooltip", [])
    max_black_3d = 2500  # 超过此数量不再渲染 3D 黑块，仅保留 2D
    black_list = list(zip(black_xs, black_ys, black_colors, black_tooltips))
    if len(black_list) <= max_black_3d:
        for px, py, fill_color, tooltip in black_list:
            g = _patch_to_geom(px, py)
            if g is not None:
                tris = _triangles_from_geom(g, simplify_tol=0.0)
                if tris:
                    mesh_black = _mesh_from_triangles(
                        tris,
                        z_func=lambda tri: _terrain_z(float(tri.centroid.x), float(tri.centroid.y)) + 10.0,
                        color=_mix_with_white(fill_color or "#cccccc", color_weight=0.22),
                    )
                    mesh_black.opacity = 0.52
                    mesh_black.hovertemplate = (tooltip if tooltip else "Black cell") + "<extra></extra>"
                    fig.add_trace(mesh_black)

            sx, sy = _smooth_closed_path(px, py, points_per_edge=10, smooth_passes=2)
            zline = [_terrain_z(float(xi), float(yi)) + 13.0 for xi, yi in zip(sx, sy)]
            fig.add_trace(
                go.Scatter3d(
                    x=sx,
                    y=sy,
                    z=zline,
                    mode="lines",
                    line=dict(color="black", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    n_black = len(black_list)
    title = f"3D 分区山体图（2-4高峰，最高{int(z_max)}m）"
    if n_black > max_black_3d:
        title += f"（黑块>{max_black_3d}，3D仅显示山体+绿框，2D可查看全部）"
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        scene=dict(
            aspectmode="data",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z",
            xaxis=dict(showbackground=True, backgroundcolor="#ffffff", gridcolor="rgba(180,180,180,0.5)"),
            yaxis=dict(showbackground=True, backgroundcolor="#ffffff", gridcolor="rgba(180,180,180,0.5)"),
            zaxis=dict(range=[0, z_max], showbackground=True, backgroundcolor="#ffffff", gridcolor="rgba(180,180,180,0.5)"),
        ),
        scene_camera=dict(eye=dict(x=1.42, y=1.42, z=0.82)),
        font=dict(color="#222222", size=11),
    )
    return fig


def _compute_black_assignment(
    cfg: GenerationConfig,
    red_gdf: gpd.GeoDataFrame,
    green_gdf: gpd.GeoDataFrame,
    bbox_geom: Polygon,
    black_gdf: gpd.GeoDataFrame,
) -> Tuple[ColumnDataSource, float]:
    """
    按质心归属：黑色块质心落在哪个红色矿物块内，即属于该块。不计算面积占比和矿物比率，加快计算。
    """
    t0 = time.perf_counter()

    centroids = gpd.GeoDataFrame(
        {"cell_id": black_gdf["cell_id"].tolist()},
        geometry=black_gdf.geometry.centroid,
        crs=None,
    )

    # 绿色区域归属
    green_join = gpd.sjoin(
        centroids,
        green_gdf[["green_id", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    green_id_by_cell = dict(zip(green_join["cell_id"], green_join["green_id"]))

    # 红色矿物块归属（质心 within）
    red_join = gpd.sjoin(
        centroids,
        red_gdf[["red_id", "FeO", "TiO", "Al2O3", "SiO2", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    miss = red_join["red_id"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(
            centroids.loc[miss],
            red_gdf[["red_id", "FeO", "TiO", "Al2O3", "SiO2", "geometry"]],
            how="left",
        ).drop(columns=["index_right"], errors="ignore")
        red_join.loc[miss, ["red_id", "FeO", "TiO", "Al2O3", "SiO2"]] = near[
            ["red_id", "FeO", "TiO", "Al2O3", "SiO2"]
        ].to_numpy()

    red_by_cell = red_join.drop_duplicates(subset=["cell_id"], keep="first").set_index("cell_id")
    tooltip_by_cell: Dict[str, str] = {}
    fill_by_cell: Dict[str, str] = {}
    black_records = black_gdf.to_dict("records")

    for rec in black_records:
        cell_id = str(rec["cell_id"])
        cell_area = float(rec["geometry"].area)
        if cell_area <= 0:
            continue

        green_id = str(green_id_by_cell.get(cell_id, ""))
        red_id = "—"
        amounts: Dict[str, float] = {}
        if cell_id in red_by_cell.index:
            r = red_by_cell.loc[cell_id]
            red_id = str(r["red_id"]) if pd.notna(r["red_id"]) else "—"
            amounts = {m: float(r.get(m, 0.0) or 0.0) for m in MINERALS}
        dominant = _dominant_mineral_from_amounts(amounts)
        fill_by_cell[cell_id] = MINERAL_COLORS.get(dominant, "#cccccc")

        tooltip_by_cell[cell_id] = (
            f"<b>{cell_id}</b>"
            f"<br>Green Zone: {green_id}"
            f"<br>Area: {cell_area:.1f} m²"
            f"<br>所属矿物块: {red_id}"
        )

    # 小格子不 simplify，加快 patch 组装
    simplify_tol_black = 0.0 if cfg.cell_size <= 20 else max(1.0, cfg.cell_size * 0.01)

    patch_x: List[List[float]] = []
    patch_y: List[List[float]] = []
    patch_tooltips: List[str] = []
    patch_fill_colors: List[str] = []

    for rec in black_records:
        cell_id = str(rec["cell_id"])
        patches = _geom_to_patches(rec["geometry"], simplify_tol=simplify_tol_black)
        if not patches:
            continue
        for px, py in patches:
            patch_x.append(px)
            patch_y.append(py)
            patch_tooltips.append(tooltip_by_cell.get(cell_id, ""))
            patch_fill_colors.append(fill_by_cell.get(cell_id, "#cccccc"))

    t1 = time.perf_counter()
    precompute_ms = (t1 - t0) * 1000.0

    cds = ColumnDataSource(
        {
            "xs": patch_x,
            "ys": patch_y,
            "tooltip": patch_tooltips,
            "fill_color": patch_fill_colors,
        }
    )
    return cds, precompute_ms


@dataclass
class UIState:
    p: object
    red_colorbar: object
    green_source: ColumnDataSource
    black_source: ColumnDataSource
    timing_div: Div


def _build_green_source(green_gdf: gpd.GeoDataFrame, cfg: GenerationConfig) -> ColumnDataSource:
    # Small-cell scenes need exact outlines; simplification can hide tiny green zones.
    simplify_tol_green = 0.0 if cfg.cell_size <= 20 else max(1.0, cfg.cell_size * 0.02)
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    for rec in green_gdf.to_dict("records"):
        patches = _geom_to_patches(rec["geometry"], simplify_tol=simplify_tol_green)
        for px, py in patches:
            xs.append(px)
            ys.append(py)
    return ColumnDataSource({"xs": xs, "ys": ys})


def modify_doc(doc) -> None:
    doc.title = "Interactive Mining Blocks (2D / 3D)"
    doc.theme = "light_minimal"

    cfg_default = GenerationConfig(
        green_area_ratio_min=0.30,
        green_area_ratio_max=0.60,
        green_zone_count_min=3,
        green_zone_count_max=3,
        seed=77,
        minerals_per_red_min=1,
        minerals_per_red_max=3,
    )

    bbox_geom = Polygon(
        [
            (cfg_default.x_min, cfg_default.y_min),
            (cfg_default.x_max, cfg_default.y_min),
            (cfg_default.x_max, cfg_default.y_max),
            (cfg_default.x_min, cfg_default.y_max),
        ]
    )

    # Figure
    p = figure(
        title="Interactive Mining Blocks (Hover on Black Cells)",
        x_range=(cfg_default.x_min, cfg_default.x_max),
        y_range=(cfg_default.y_min, cfg_default.y_max),
        match_aspect=True,
        width=1040,
        height=820,
        tools="pan,wheel_zoom,reset,save",
    )
    p.title.align = "center"
    p.grid.grid_line_alpha = 0.2

    # Green outline only
    green_source = ColumnDataSource({"xs": [], "ys": []})
    green_renderer = p.patches(
        "xs",
        "ys",
        source=green_source,
        fill_alpha=0.0,
        line_color="#2bbf4b",
        line_width=2.4,
        level="overlay",
    )

    # Red mineral base polygons on the map (no hover; legend still shows icons).
    red_source = ColumnDataSource({"xs": [], "ys": [], "fill_color": []})
    red_renderer = p.patches(
        "xs",
        "ys",
        source=red_source,
        fill_color={"field": "fill_color"},
        fill_alpha=1.0,
        line_color="#b00020",
        line_alpha=0.25,
        line_width=1.0,
    )

    # Black cells
    black_source = ColumnDataSource({"xs": [], "ys": [], "tooltip": [], "fill_color": []})
    black_renderer = p.patches(
        "xs",
        "ys",
        source=black_source,
        fill_color={"field": "fill_color"},
        fill_alpha=0.30,
        line_color="black",
        line_alpha=0.65,
        line_width=0.6,
    )
    p.add_tools(HoverTool(tooltips=[("Details", "@tooltip{safe}")], renderers=[black_renderer]))

    # Timing
    timing_div = Div(
        text="",
        width=320,
        height=220,
    )

    # Red blocks legend (icons only; hover is enabled only for black cells).
    legend_div = Div(
        text="",
        width=320,
        height=620,
    )

    # Controls
    green_num = Spinner(title="Green Zones Num", low=1, high=6, step=1, value=2)
    cell_size_m = Spinner(title="Black Cell (gen) m", low=5, high=600, step=5, value=55)

    def _set_timing(precompute_ms: float, patch_ms: float, total_s: float) -> None:
        # show ms vs s clearly
        if precompute_ms >= 1000:
            unit = "s"
            precompute_display = precompute_ms / 1000.0
            timing_line = f"归属计算：{precompute_display:.2f} s（总预计算）"
        else:
            unit = "ms"
            timing_line = f"归属计算：{precompute_ms:.0f} ms（总预计算）"

        timing_div.text = (
            f"<b>归属计算耗时</b><br/>"
            f"{timing_line}<br/>"
            f"patch 组装：{patch_ms:.0f} ms<br/>"
            f"总耗时：{total_s:.2f} s"
        )

    # 缓存：仅 Green Zones Num 变化时重新生成红/绿；cell_size 变化时复用红/绿，并缓存 black_gdf
    _cache: Dict[str, object] = {
        "red_gdf": None,
        "green_gdf": None,
        "bbox": None,
        "green_num": -1,
        "green_union": None,
        "black_by_cell": {},  # cell_size -> black_gdf
    }

    def _render(cfg: GenerationConfig) -> None:
        t0 = time.perf_counter()
        gnum = int(cfg.green_zone_count_min)
        csize = int(cfg.cell_size)

        if _cache["green_num"] != gnum:
            red_gdf, green_gdf, bbox_geom2 = generate_red_and_green(cfg)
            _cache["red_gdf"] = red_gdf
            _cache["green_gdf"] = green_gdf
            _cache["bbox"] = bbox_geom2
            _cache["green_num"] = gnum
            _cache["green_union"] = unary_union(green_gdf.geometry.tolist()).buffer(0)
            _cache["black_by_cell"] = {}
        else:
            red_gdf = _cache["red_gdf"]
            green_gdf = _cache["green_gdf"]
            bbox_geom2 = _cache["bbox"]

        green_union = _cache["green_union"]
        if csize in _cache["black_by_cell"]:
            black_gdf = _cache["black_by_cell"][csize]
        else:
            black_gdf = build_clipped_grid(green_geom=green_union, bbox_geom=bbox_geom2, cell_size=csize)
            _cache["black_by_cell"][csize] = black_gdf

        # update green source
        green_source.data = dict(_build_green_source(green_gdf, cfg).data)

        # update red mineral base polygons
        try:
            simplify_tol_red = max(1.0, cfg.cell_size * 0.01)
            red_xs: List[List[float]] = []
            red_ys: List[List[float]] = []
            red_colors: List[str] = []
            for rec in red_gdf.to_dict("records"):
                geom = rec["geometry"]
                dominant = _dominant_mineral_from_amounts({m: float(rec.get(m, 0.0) or 0.0) for m in MINERALS})
                # Use pre-lightened opaque colors to avoid false "overlap" perception from alpha stacking.
                fill = _mix_with_white(MINERAL_COLORS.get(dominant, "#cccccc"), color_weight=0.22)
                patches = _geom_to_patches(geom, simplify_tol=simplify_tol_red)
                for px, py in patches:
                    red_xs.append(px)
                    red_ys.append(py)
                    red_colors.append(fill)
            red_source.data = dict(xs=red_xs, ys=red_ys, fill_color=red_colors)
        except Exception:
            # Keep previous red layer if update fails.
            pass

        # Update red legend (area ratio + minerals for each red block).
        try:
            bbox_area = float(bbox_geom2.area)
            legend_items: List[str] = []
            for rec in red_gdf.to_dict("records"):
                rid = str(rec.get("red_id"))
                minerals_str = str(rec.get("minerals", "") or "").strip()
                if not minerals_str:
                    # Fallback: build minerals string from columns with positive values
                    minerals_str = ", ".join([m for m in MINERALS if float(rec.get(m, 0.0) or 0.0) > 0])
                area_ratio = (float(rec["geometry"].area) / bbox_area * 100.0) if bbox_area > 0 else 0.0
                # Determine dot color by dominant mineral amount.
                dominant = _dominant_mineral_from_amounts({m: float(rec.get(m, 0.0) or 0.0) for m in MINERALS})
                dot_color = MINERAL_COLORS.get(dominant, "#999999")
                legend_items.append(
                    f"<div style='display:flex;gap:8px;align-items:flex-start;margin:4px 0;'>"
                    f"<span style='width:10px;height:10px;background:{dot_color};border:1px solid rgba(0,0,0,0.3);margin-top:3px;flex:0 0 auto;'></span>"
                    f"<div style='line-height:1.25;'>"
                    f"<div style='font-size:12px;'><b>{rid}</b> — {area_ratio:.2f}% of Bounding Box</div>"
                    f"<div style='font-size:11px;color:#333;margin-top:1px;'>Minerals: {minerals_str if minerals_str else '—'}</div>"
                    f"</div></div>"
                )
            legend_div.text = (
                "<div style='font-size:12px;'><b>Red Blocks Legend</b><br/>(icons only; red polygons are shown on map, but no red hover)</div>"
                f"<div style='margin-top:8px;max-height:560px;overflow:auto;padding-right:6px;'>{''.join(legend_items) if legend_items else '<div>No red blocks</div>'}</div>"
            )
        except Exception as e:
            legend_div.text = f"<div style='color:#b00020;font-size:12px;'>Legend failed: {type(e).__name__}</div>"

        # compute black assignment + prepare patches
        cds, precompute_ms = _compute_black_assignment(cfg, red_gdf, green_gdf, bbox_geom2, black_gdf)
        black_source.data = dict(cds.data)

        # update 3D tab
        _update_3d(red_gdf, green_gdf, black_gdf, bbox_geom2, cfg)

        # measure patch ms approximately
        patch_ms = precompute_ms  # in this simplified version, cds build includes patch arrays
        total_s = time.perf_counter() - t0

        _set_timing(precompute_ms=precompute_ms, patch_ms=patch_ms, total_s=total_s)

    def _on_apply():
        try:
            w, h = 100, 100  # 与 generator 默认 x_max-x_min, y_max-y_min 一致
            n_red = (6, 10) if w * h <= 20000 else (28, 34)
            min_edge = 2.0 if w * h <= 20000 else 90.0
            cfg = GenerationConfig(
                red_count_min=n_red[0],
                red_count_max=n_red[1],
                min_shared_edge_m=min_edge,
                green_area_ratio_min=0.05,
                green_area_ratio_max=0.45,
                green_zone_count_min=int(green_num.value),
                green_zone_count_max=int(green_num.value),
                cell_size=int(cell_size_m.value),
                seed=77,
                minerals_per_red_min=1,
                minerals_per_red_max=3,
            )
            _render(cfg)
        except Exception as e:
            timing_div.text = f"<b>生成失败</b><br/><span style='color:#b00020'>{type(e).__name__}: {e}</span>"

    # Debounced regen on spinner change (optional)
    regen_delay_ms = 250
    timeout_cb_id: Optional[str] = None

    def _schedule_regen(attr, old, new):
        nonlocal timeout_cb_id
        if timeout_cb_id is not None:
            try:
                doc.remove_timeout_callback(timeout_cb_id)
            except Exception:
                pass
        timeout_cb_id = doc.add_timeout_callback(lambda: _on_apply(), regen_delay_ms)

    green_num.on_change("value", _schedule_regen)
    cell_size_m.on_change("value", _schedule_regen)

    # 3D tab: 导出独立 HTML，用 iframe 加载（避免 Bokeh 内嵌 Plotly 不显示）
    _output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(_output_dir, exist_ok=True)
    _3d_html_path = os.path.join(_output_dir, "3d_preview.html")

    # 为了让黑色显示框与 2D 图框同尺度，iframe 内 Plotly 也用固定尺寸渲染（不要依赖 responsive）。
    plotly_container = Div(
        text=f'<iframe id="plotly-3d-iframe" src="/output/3d_preview.html" style="width:1040px;height:820px;min-height:820px;border:none;background:#ffffff;display:block;"></iframe>',
        width=1040,
        height=820,
    )

    def _update_3d(red_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame, black_gdf: gpd.GeoDataFrame, bbox_geom: Polygon, cfg: GenerationConfig) -> None:
        try:
            fig = _build_3d_figure(
                red_data=dict(red_source.data),
                green_data=dict(green_source.data),
                black_data=dict(black_source.data),
                cfg=cfg,
            )
            # 每次写入不同文件名，强制与当前2D结果同步，彻底规避浏览器缓存
            ts = int(time.time() * 1000)
            out_name = f"3d_preview_{ts}.html"
            out_path = os.path.join(_output_dir, out_name)
            # 固定导出尺寸，避免 iframe 内部自适应导致显示框变小
            pio.write_html(
                fig,
                out_path,
                config={"responsive": False},
                default_width=1040,
                default_height=820,
                auto_play=False,
            )
            latest_3d_url = f"/output/{out_name}"
            plotly_container.text = (
                f'<iframe id="plotly-3d-iframe" src="{latest_3d_url}" '
                'style="width:1040px;height:820px;min-height:820px;border:none;background:#ffffff;display:block;"></iframe>'
            )
        except Exception as e:
            # 不再静默吞错，避免“代码改了但看起来没生效”
            timing_div.text = (
                timing_div.text
                + f"<br/><span style='color:#b00020;'><b>3D failed:</b> {type(e).__name__}: {e}</span>"
            )
            err_html = (
                "<html><body style='margin:20px;font-family:sans-serif;background:#1e1f23;color:#f4f4f4;'>"
                f"<h3>3D render failed</h3><pre>{type(e).__name__}: {e}</pre></body></html>"
            )
            try:
                with open(_3d_html_path, "w", encoding="utf-8") as f:
                    f.write(err_html)
                ts = int(time.time() * 1000)
                latest_3d_url = f"/output/3d_preview.html?t={ts}"
                plotly_container.text = (
                    f'<iframe id="plotly-3d-iframe" src="{latest_3d_url}" '
                    'style="width:1040px;height:820px;min-height:820px;border:none;background:#ffffff;display:block;"></iframe>'
                )
            except Exception:
                pass

    # Layout: 2D + 3D tabs
    title_div = Div(
        text="<div style='text-align:center;font-size:20px;font-weight:700;color:#222;padding:8px 0;'>Interactive Mining Blocks (2D / 3D)</div>",
        width=1360,
        height=40,
    )

    controls_col = column(green_num, cell_size_m, width=220)
    sidebar_col = column(timing_div, legend_div, width=320)

    tabs = Tabs(
        tabs=[
            ("2D", p),
            ("3D 山体", plotly_container),
        ],
        width=1040,
        height=1100,
    )
    # 3D iframe 现在由 Python 侧直接重写，不依赖前端回调更新 URL。

    top_row = row(controls_col)
    content_row = row(tabs, sidebar_col)
    layout = column(title_div, top_row, content_row, align="center", sizing_mode="fixed", width=1360, height=1140)
    doc.add_root(layout)

    # initial render
    _on_apply()


if __name__ == "__main__":
    import argparse
    from tornado.web import StaticFileHandler

    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from bokeh.server.server import Server

    parser = argparse.ArgumentParser(description="Interactive Mining Blocks (2D/3D)")
    parser.add_argument("--port", type=int, default=5006, help="Bokeh server port")
    parser.add_argument("--show", action="store_true", help="Open browser after start")
    args = parser.parse_args()

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _out_dir = os.path.join(_script_dir, "output")
    os.makedirs(_out_dir, exist_ok=True)
    _preview_path = os.path.join(_out_dir, "3d_preview.html")
    if not os.path.exists(_preview_path):
        with open(_preview_path, "w", encoding="utf-8") as f:
            f.write("<html><body style='margin:20px;font:16px sans-serif;'>点击 Regenerate/Apply 后切换到 3D 标签查看</body></html>")
    extra_patterns = [
        (r"/output/(.*)", StaticFileHandler, {"path": _out_dir}),
    ]

    apps = {"/": Application(FunctionHandler(modify_doc))}
    server = Server(apps, port=args.port, allow_websocket_origin=["*"], extra_patterns=extra_patterns)
    server.start()
    print(f"Running: http://localhost:{args.port}/")
    if args.show:
        server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
else:
    modify_doc(curdoc())

