from __future__ import annotations

import time
from typing import Dict, List, Tuple

import geopandas as gpd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColorBar, ColumnDataSource, Div, HoverTool, LinearColorMapper
from bokeh.palettes import Reds
from bokeh.plotting import figure
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from generator import GenerationConfig, generate_red_and_green
from grid_assign import build_clipped_grid

MINERALS = ["FeO", "TiO", "Al2O3", "SiO2"]


def _geom_to_patches(geom, simplify_tol: float) -> List[Tuple[List[float], List[float]]]:
    """
    Convert a (Multi)Polygon geometry into Bokeh patch rings (exterior only).
    """
    if geom is None or geom.is_empty:
        return []

    geom2 = geom
    try:
        geom2 = geom2.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        pass

    if isinstance(geom2, Polygon):
        x, y = geom2.exterior.xy
        return [(list(x), list(y))]

    if isinstance(geom2, MultiPolygon):
        patches: List[Tuple[List[float], List[float]]] = []
        for g in geom2.geoms:
            patches.extend(_geom_to_patches(g, simplify_tol))
        return patches

    return []


def _format_minerals_for_red(row: Dict) -> str:
    parts: List[str] = []
    for m in MINERALS:
        v = float(row.get(m, 0.0) or 0.0)
        if v > 0:
            parts.append(f"{m}={v:.2f}")
    return ", ".join(parts) if parts else "None"


def generate_interactive_html(
    cfg: GenerationConfig,
    out_html: str,
    red_gdf=None,
    green_gdf=None,
    bbox_geom=None,
    black_grid=None,
) -> None:
    """
    Standalone interactive HTML (open in browser):
    - Hover a red mining block: show its mineral list (1-3 kinds).
    - Hover a black cell: show which red blocks it covers + area-weighted mineral shares.
    """
    t0 = time.perf_counter()

    if red_gdf is None or green_gdf is None or bbox_geom is None:
        red_gdf, green_gdf, bbox_geom = generate_red_and_green(cfg)

    if black_grid is None:
        green_union = unary_union(green_gdf.geometry.tolist()).buffer(0)
        black_grid = build_clipped_grid(green_geom=green_union, bbox_geom=bbox_geom, cell_size=cfg.cell_size)

    # Assign green_id to each black cell by centroid.
    centroids = gpd.GeoDataFrame(
        {"cell_id": black_grid["cell_id"].tolist()},
        geometry=black_grid.geometry.centroid,
        crs=None,
    )
    green_join = gpd.sjoin(
        centroids,
        green_gdf[["green_id", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    green_id_by_cell = dict(zip(green_join["cell_id"], green_join["green_id"]))

    # Spatial index for red blocks.
    red_sindex = red_gdf.sindex
    red_polys = red_gdf.geometry.tolist()
    red_records = red_gdf.to_dict("records")

    mineral_colors = {
        "FeO": "#d73027",
        "TiO": "#4575b4",
        "Al2O3": "#1a9850",
        "SiO2": "#984ea3",
    }

    # Precompute tooltip + fill-color per black cell.
    cell_tooltip_by_id: Dict[str, str] = {}
    cell_fill_by_id: Dict[str, str] = {}

    simplify_tol_black = max(1.0, cfg.cell_size * 0.01)
    simplify_tol_red = max(0.8, cfg.cell_size * 0.005)

    black_records = black_grid.to_dict("records")
    t_precompute_black_start = time.perf_counter()

    for rec in black_records:
        cell_id = str(rec["cell_id"])
        cell_geom = rec["geometry"]
        cell_area = float(cell_geom.area)
        if cell_area <= 0:
            continue

        green_id = green_id_by_cell.get(cell_id, "")

        overlaps: List[Tuple[str, float]] = []
        mineral_amounts = {m: 0.0 for m in MINERALS}

        candidates = list(red_sindex.intersection(cell_geom.bounds))
        for ridx in candidates:
            inter_area = float(cell_geom.intersection(red_polys[ridx]).area)
            if inter_area <= 1e-6:
                continue

            red_row = red_records[ridx]
            red_id = str(red_row["red_id"])
            prop = inter_area / cell_area
            overlaps.append((red_id, prop))

            for m in MINERALS:
                mineral_amounts[m] += prop * float(red_row.get(m, 0.0) or 0.0)

        total_amount = sum(mineral_amounts.values())
        mineral_shares = {
            m: (mineral_amounts[m] / total_amount if total_amount > 0 else 0.0) for m in MINERALS
        }

        overlaps_sorted = sorted(overlaps, key=lambda x: x[1], reverse=True)
        overlaps_lines = (
            "<br>".join([f"{rid} ({p * 100:.1f}%)" for rid, p in overlaps_sorted[:6]])
            if overlaps_sorted
            else "None"
        )

        minerals_sorted = sorted(
            [(m, s) for m, s in mineral_shares.items() if s > 1e-6], key=lambda x: x[1], reverse=True
        )
        minerals_lines = (
            "<br>".join([f"{m} ({s * 100:.1f}%)" for m, s in minerals_sorted[:6]])
            if minerals_sorted
            else "None"
        )

        dominant = minerals_sorted[0][0] if minerals_sorted else "FeO"
        cell_fill_by_id[cell_id] = mineral_colors.get(dominant, "#cccccc")

        tooltip = (
            f"<b>{cell_id}</b>"
            f"<br>Green Zone: {green_id}"
            f"<br>Black cell area: {cell_area:.1f} m²"
            f"<br><br><b>Covered red blocks</b><br>{overlaps_lines}"
            f"<br><br><b>Minerals share (area-weighted)</b><br>{minerals_lines}"
        )
        cell_tooltip_by_id[cell_id] = tooltip

    t_precompute_black_ms = (time.perf_counter() - t_precompute_black_start) * 1000.0

    # Build patch-level datasource (explode MultiPolygon -> multiple patches).
    t_patch_build_start = time.perf_counter()
    patch_x: List[List[float]] = []
    patch_y: List[List[float]] = []
    patch_tooltips: List[str] = []
    patch_fill_colors: List[str] = []

    for rec in black_records:
        cell_id = str(rec["cell_id"])
        patches = _geom_to_patches(rec["geometry"], simplify_tol=simplify_tol_black)
        for px, py in patches:
            patch_x.append(px)
            patch_y.append(py)
            patch_tooltips.append(cell_tooltip_by_id.get(cell_id, ""))
            patch_fill_colors.append(cell_fill_by_id.get(cell_id, "#cccccc"))

    # Red patches (render underlay).
    red_patch_x: List[List[float]] = []
    red_patch_y: List[List[float]] = []
    color_values: List[float] = []

    color_vals = [float(r.get("color_value", 0.0) or 0.0) for r in red_records]
    color_min = min(color_vals) if color_vals else 0.0
    color_max = max(color_vals) if color_vals else 1.0

    for rec in red_records:
        patches = _geom_to_patches(rec["geometry"], simplify_tol=simplify_tol_red)
        cv = float(rec.get("color_value", 0.0) or 0.0)
        for px, py in patches:
            red_patch_x.append(px)
            red_patch_y.append(py)
            color_values.append(cv)

    # Green outlines.
    green_patch_x: List[List[float]] = []
    green_patch_y: List[List[float]] = []
    for rec in green_gdf.to_dict("records"):
        patches = _geom_to_patches(rec["geometry"], simplify_tol=simplify_tol_red)
        for px, py in patches:
            green_patch_x.append(px)
            green_patch_y.append(py)

    # Red hover is disabled in this page; tooltip field is not required.
    red_source = ColumnDataSource({"xs": red_patch_x, "ys": red_patch_y, "color_value": color_values})
    black_source = ColumnDataSource({"xs": patch_x, "ys": patch_y, "tooltip": patch_tooltips, "fill_color": patch_fill_colors})
    green_source = ColumnDataSource({"xs": green_patch_x, "ys": green_patch_y})
    t_patch_build_ms = (time.perf_counter() - t_patch_build_start) * 1000.0

    # Build figure.
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = bbox_geom.bounds
    plot_width = 1040
    plot_height = 820
    p = figure(
        title="Interactive Mining Blocks (Hover for details)",
        x_range=(bbox_minx, bbox_maxx),
        y_range=(bbox_miny, bbox_maxy),
        match_aspect=True,
        width=plot_width,
        height=plot_height,
        tools="pan,wheel_zoom,reset,save",
    )
    p.grid.grid_line_alpha = 0.2
    p.title.text_font_size = "13pt"

    mapper = LinearColorMapper(palette=Reds[6], low=color_min, high=color_max)
    red_renderer = p.patches(
        "xs",
        "ys",
        source=red_source,
        fill_color={"field": "color_value", "transform": mapper},
        fill_alpha=0.62,
        line_color="#8b0000",
        line_width=1.0,
    )
    p.add_layout(ColorBar(color_mapper=mapper, location=(0, 0)))

    p.patches(
        "xs",
        "ys",
        source=green_source,
        fill_alpha=0.0,
        line_color="#2bbf4b",
        line_width=2.0,
    )

    black_renderer = p.patches(
        "xs",
        "ys",
        source=black_source,
        fill_color={"field": "fill_color"},
        fill_alpha=0.28,
        line_color="black",
        line_alpha=0.65,
        line_width=0.6,
    )

    # Only keep black-cell hover. Remove red-cell hover to meet your requirement.
    p.add_tools(HoverTool(tooltips=[("Details", "@tooltip{safe}")], renderers=[black_renderer]))

    # ---- Right-side legend + algorithm + timing ----
    bbox_area = float(bbox_geom.area)
    green_zone_count = len(green_gdf)
    green_area = float(unary_union(green_gdf.geometry.tolist()).area) if green_zone_count else 0.0
    black_area = float(sum(float(r["geometry"].area) for r in black_records)) if black_records else 0.0

    # Map red blocks to colors used in plot for legend swatches.
    def _val_to_palette_color(v: float) -> str:
        if color_max <= color_min:
            idx = len(Reds[6]) - 1
        else:
            t = (v - color_min) / (color_max - color_min)
            idx = max(0, min(len(Reds[6]) - 1, int(t * (len(Reds[6]) - 1) + 0.5)))
        return Reds[6][idx]

    red_items: List[str] = []
    for rec in red_records:
        rid = str(rec["red_id"])
        area_ratio = float(rec["geometry"].area) / bbox_area if bbox_area > 0 else 0.0
        minerals_str = str(rec.get("minerals", "") or "").strip()
        cv = float(rec.get("color_value", 0.0) or 0.0)
        sw = _val_to_palette_color(cv)
        red_items.append(
            f"<div style='display:flex;gap:8px;align-items:flex-start;margin:6px 0;'>"
            f"<span style='width:12px;height:12px;background:{sw};border:1px solid rgba(0,0,0,0.3);flex:0 0 auto;margin-top:3px;'></span>"
            f"<div style='line-height:1.25;'>"
            f"<div style='font-size:12px;'><b>{rid}</b> — {area_ratio*100:.2f}% of Bounding Box</div>"
            f"<div style='font-size:11px;color:#333;margin-top:2px;'>Minerals: {minerals_str if minerals_str else '—'}</div>"
            f"</div></div>"
        )

    red_legend_html = "".join(red_items) if red_items else "<div>No red blocks</div>"

    elapsed_total = time.perf_counter() - t0

    algorithm_html = (
        "<div style='font-size:12px;line-height:1.4;'>"
        "<b>黑色单元所属红色块的算法</b><br/>"
        "对每个黑色单元多边形，使用红色块空间索引（GeoPandas sindex/rtree）筛选可能相交的红色块；"
        "逐个计算交集面积 <i>inter_area</i>，占比 <i>prop = inter_area / cell_area</i>；"
        "再按占比对该红色块矿物含量做面积加权，得到该黑色单元的矿物占比，并列出覆盖的红色块。<br/>"
        "<b>关键点</b>：所有交集计算都在 Python 生成阶段预计算写入 HTML，所以鼠标移动时不再做 GIS 空间计算（hover 基本毫秒级）。"
        "</div>"
    )

    timing_html = (
        f"<div style='font-size:12px;line-height:1.4;margin-top:8px;'>"
        f"<b>耗时统计</b><br/>"
        f"预计算黑色单元属性：{t_precompute_black_ms:.0f} ms<br/>"
        f"几何转 patch + 组装渲染数据：{t_patch_build_ms:.0f} ms<br/>"
        f"总耗时：{elapsed_total:.2f} s<br/>"
        f"</div>"
    )

    parameter_row_html = (
        f"<div style='font-size:12px;display:flex;gap:16px;align-items:flex-start;justify-content:space-between;'>"
        f"<div><b>Green Zones</b><br/>{green_zone_count}</div>"
        f"<div><b>Green Total Area</b><br/>{green_area/1e6:.3f} km²</div>"
        f"<div><b>Black Total Area</b><br/>{black_area/1e6:.3f} km²</div>"
        f"<div><b>Black Cell (gen)</b><br/>{cfg.cell_size} m</div>"
        "</div>"
    )

    title_div = Div(
        text="<div style='font-size:20px;font-weight:700;color:#222;padding:6px 14px 0 14px;'>Interactive Mining Blocks Dashboard</div>",
        width=plot_width + 320,
        height=34,
    )

    params_div = Div(
        text=f"<div style='background:#f6f7fb;border:1px solid #e4e7ef;border-radius:10px;padding:10px 14px;'>{parameter_row_html}</div>",
        width=plot_width + 320,
        height=70,
    )

    legend_div = Div(
        text=(
            "<div style='font-size:12px;'>"
            "<b>Red Blocks Legend</b><br/>"
            "（不显示红色块编号 Hover，只在此处汇总）"
            "</div>"
            f"<div style='margin-top:8px;max-height:820px;overflow:auto;padding-right:4px;'>{red_legend_html}</div>"
            f"{algorithm_html}"
            f"{timing_html}"
        ),
        width=320,
        height=plot_height + 30,
    )

    sidebar = column(legend_div, width=320, sizing_mode="fixed")
    layout = column(title_div, params_div, row(p, sidebar, sizing_mode="fixed"), sizing_mode="fixed")

    # Generate HTML.
    output_file(out_html, title="Interactive Mining Blocks")
    save(layout)

    # Post-process HTML to center content.
    try:
        with open(out_html, "r", encoding="utf-8") as f:
            html_txt = f.read()
        if "display: flow-root;" in html_txt:
            html_txt = html_txt.replace(
                "display: flow-root;",
                "display:flex; justify-content:center; align-items:flex-start;",
            )
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html_txt)
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    print(f"[interactive] black cells: {len(black_grid)}")
    print(f"[interactive] generated: {out_html}")
    print(f"[interactive] total time: {elapsed:.2f}s")

