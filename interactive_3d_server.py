from __future__ import annotations

import argparse
import time
from typing import Dict, Iterable, List

import plotly.graph_objects as go
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, CustomJS, ColumnDataSource, Div, Spinner
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union, triangulate
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.server.server import Server

from generator import GenerationConfig, generate_red_and_green
from grid_assign import build_clipped_grid, assign_properties_to_cells


MINERALS = ["FeO", "TiO", "Al2O3", "SiO2"]
MINERAL_COLORS = {
    "FeO": "#d73027",
    "TiO": "#4575b4",
    "Al2O3": "#1a9850",
    "SiO2": "#984ea3",
}


def _dominant_mineral_from_amounts(amounts: Dict[str, float]) -> str:
    best = "FeO"
    best_v = -1.0
    for m in MINERALS:
        v = float(amounts.get(m, 0.0) or 0.0)
        if v > best_v:
            best_v = v
            best = m
    return best


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
    # Shapely may return list or GeometryCollection; support both
    iter_tris = getattr(tris, "geoms", tris) if tris is not None else []
    for t in iter_tris:
        if not isinstance(t, Polygon):
            continue
        if t.area <= 1e-9:
            continue
        # keep only triangles inside polygon
        if t.centroid.within(p.buffer(1e-9)):
            out.append(t)
    return out


def _triangles_from_geom(geom, simplify_tol: float) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return _triangles_from_polygon(geom, simplify_tol)
    if isinstance(geom, MultiPolygon):
        tris: List[Polygon] = []
        for g in geom.geoms:
            tris.extend(_triangles_from_polygon(g, simplify_tol))
        return tris
    return []


def _mesh_from_triangles(
    triangles: Iterable[Polygon],
    z_func,
    color: str,
) -> go.Mesh3d:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    i: List[int] = []
    j: List[int] = []
    k: List[int] = []

    tri_count = 0
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
        tri_count += 1

    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.95,
        flatshading=True,
        name="mesh",
    )


def build_mountain_fig(
    cell_size: int,
    green_zone_count: int,
    seed: int = 77,
    base_amp: float = 20.0,
    top_amp: float = 60.0,
    triangulate_tol_mult_red: float = 0.03,
    triangulate_tol_mult_black: float = 0.02,
) -> tuple[go.Figure, Dict[str, float]]:
    """
    3D mountain:
    - Red polygons are mountain base.
    - Black grid cells are the top surface relief.
    - Height is derived from mineral totals on the black cell (area-weighted by assignment).
    """
    cfg = GenerationConfig(
        red_count_min=28,
        red_count_max=34,
        min_shared_edge_m=90,
        green_area_ratio_min=0.30,
        green_area_ratio_max=0.60,
        green_zone_count_min=green_zone_count,
        green_zone_count_max=green_zone_count,
        cell_size=cell_size,
        seed=seed,
        minerals_per_red_min=1,
        minerals_per_red_max=3,
    )

    t0 = time.perf_counter()
    red_gdf, green_gdf, bbox_geom = generate_red_and_green(cfg)
    green_union = unary_union(green_gdf.geometry.tolist()).buffer(0)
    black_grid = build_clipped_grid(green_geom=green_union, bbox_geom=bbox_geom, cell_size=cfg.cell_size)
    black_gdf = assign_properties_to_cells(black_grid, red_gdf)

    timings: Dict[str, float] = {}
    timings["gen_2d_s"] = time.perf_counter() - t0

    # base z per red polygon (stable)
    red_records = red_gdf.to_dict("records")
    # use color_value as proxy
    cv = [float(r.get("color_value", 0.0) or 0.0) for r in red_records]
    cmin = min(cv) if cv else 0.0
    cmax = max(cv) if cv else 1.0
    denom = (cmax - cmin) if (cmax - cmin) != 0 else 1.0

    z_base_by_red: Dict[str, float] = {}
    dom_by_red: Dict[str, str] = {}
    for r in red_records:
        rid = str(r["red_id"])
        t = (float(r.get("color_value", 0.0) or 0.0) - cmin) / denom
        z_base_by_red[rid] = base_amp * (0.25 + 0.75 * t)
        dom_by_red[rid] = _dominant_mineral_from_amounts({m: float(r.get(m, 0.0) or 0.0) for m in MINERALS})

    # top z per black cell (based on total mineral)
    black_records = black_gdf.to_dict("records")
    totals = [_total_mineral(r) for r in black_records]
    tot_min = min(totals) if totals else 0.0
    tot_max = max(totals) if totals else 1.0
    tot_denom = (tot_max - tot_min) if (tot_max - tot_min) != 0 else 1.0

    def z_top_for_cell(cell: Dict) -> float:
        rid = str(cell["red_id"])
        zb = z_base_by_red.get(rid, base_amp * 0.25)
        t = (_total_mineral(cell) - tot_min) / tot_denom
        return zb + top_amp * t

    fig = go.Figure()

    tri_tol_red = max(1.0, cell_size * triangulate_tol_mult_red)
    tri_tol_black = max(1.0, cell_size * triangulate_tol_mult_black)

    base_tri_total = 0
    # Build base meshes (one per red polygon; may be heavy but acceptable for small red count)
    for r in red_records:
        rid = str(r["red_id"])
        dom = dom_by_red.get(rid, "FeO")
        color = MINERAL_COLORS.get(dom, "#999999")
        z_base = z_base_by_red.get(rid, base_amp * 0.25)
        tris = _triangles_from_geom(r["geometry"], tri_tol_red)
        base_tri_total += len(tris)
        mesh = _mesh_from_triangles(tris, z_func=lambda _tri, z=z_base: z, color=color)
        mesh.name = f"base_{rid}"
        fig.add_trace(mesh)

    top_tri_total = 0
    # Build top meshes (one per black cell)
    # For performance: skip very small areas
    min_black_area = (cell_size * cell_size) * 0.02
    for cell in black_records:
        cell_area = float(cell.get("cell_area", 0.0) or 0.0)
        if cell_area < min_black_area:
            continue
        dom = _dominant_mineral_from_amounts({m: float(cell.get(m, 0.0) or 0.0) for m in MINERALS})
        color = MINERAL_COLORS.get(dom, "#cccccc")
        z_top = z_top_for_cell(cell)
        tris = _triangles_from_geom(cell["geometry"], tri_tol_black)
        top_tri_total += len(tris)
        mesh = _mesh_from_triangles(tris, z_func=lambda _tri, z=z_top: z, color=color)
        mesh.name = f"top_{cell['cell_id']}"
        fig.add_trace(mesh)

    # style
    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = bbox_geom.bounds
    fig.update_layout(
        title=f"3D Mountain Mesh (cell_size={cell_size}m, green_zones={green_zone_count})",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (height)",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        scene_camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
    )

    timings["base_tris"] = float(base_tri_total)
    timings["top_tris"] = float(top_tri_total)
    timings["cell_kept"] = float(len([c for c in black_records if float(c.get("cell_area", 0.0) or 0.0) >= min_black_area]))
    return fig, timings


def _fig_to_plotly_json(fig: go.Figure) -> str:
    # Use Plotly JSON so we can render in browser via CustomJS (no script injection).
    return fig.to_json()


def modify_doc(doc) -> None:
    doc.title = "3D Mountain Dashboard"
    doc.theme = "light_minimal"

    # Controls
    green_num = Spinner(title="Green Zones Num", low=1, high=6, step=1, value=3)
    cell_size_m = Spinner(title="Black Cell (gen) m", low=100, high=600, step=50, value=300)
    apply_btn = Button(label="Generate 3D", button_type="primary", width=180)

    status = Div(text="", width=420, height=40)

    plot_container = Div(
        text="<div id='plotly-3d-container' style='width:900px;height:650px;'></div>",
        width=920,
        height=680,
    )
    plot_source = ColumnDataSource(data={"plotly_json": [""]})

    plot_source.js_on_change(
        "data",
        CustomJS(
            args=dict(src=plot_source),
            code="""
            function ensurePlotly(callback) {
              if (window.Plotly && window.Plotly.react) {
                callback();
                return;
              }

              if (window._plotlyLoadPromise) {
                window._plotlyLoadPromise.then(callback);
                return;
              }

              window._plotlyLoadPromise = new Promise(function(resolve, reject) {
                var s = document.createElement('script');
                s.src = 'https://cdn.plot.ly/plotly-2.30.0.min.js';
                s.onload = function() { resolve(); };
                s.onerror = function() { reject(new Error('Failed to load Plotly.js')); };
                document.head.appendChild(s);
              });

              window._plotlyLoadPromise.then(callback);
            }

            ensurePlotly(function() {
              var el = document.getElementById('plotly-3d-container');
              if (!el) return;
              var figJson = src.data.plotly_json[0];
              if (!figJson) return;
              var fig = JSON.parse(figJson);
              Plotly.react(el, fig.data, fig.layout, {});
            });
            """,
        ),
    )

    def _render():
        try:
            status.text = "<div style='font-size:13px;'>Generating 3D mesh... (this may take a few seconds)</div>"
            apply_btn.disabled = True

            cell_size = int(cell_size_m.value)
            green_zone_count = int(green_num.value)

            t0 = time.perf_counter()
            fig, timings = build_mountain_fig(cell_size=cell_size, green_zone_count=green_zone_count)
            elapsed_s = time.perf_counter() - t0
            plot_source.data = {"plotly_json": [_fig_to_plotly_json(fig)]}

            # timing report
            pre_ms = timings.get("gen_2d_s", elapsed_s) * 1000.0
            status.text = (
                f"<div style='font-size:12px;line-height:1.4;'>"
                f"<b>3D Render</b><br/>"
                f"cell_size={cell_size}m, green_zones={green_zone_count}<br/>"
                f"2D+assignment: {pre_ms:.0f} ms<br/>"
                f"base_tris={int(timings.get('base_tris', 0))}, top_tris={int(timings.get('top_tris', 0))}<br/>"
                f"total: {elapsed_s:.2f} s"
                f"</div>"
            )
        except Exception as e:
            status.text = f"<div style='color:#b00020;font-size:13px;'><b>Failed:</b> {type(e).__name__}: {e}</div>"
        finally:
            apply_btn.disabled = False

    def _on_apply():
        _render()

    apply_btn.on_click(_on_apply)

    layout = column(
        Div(text="<h2 style='text-align:center;margin:10px 0 0 0;'>3D Mountain Dashboard</h2>", width=960),
        row(green_num, cell_size_m, apply_btn, sizing_mode="fixed"),
        status,
        plot_container,
        sizing_mode="fixed",
    )
    doc.add_root(layout)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Mountain Dashboard (Plotly+Shapely) running as a Bokeh server.")
    parser.add_argument("--port", type=int, default=5011, help="Bokeh server port.")
    parser.add_argument("--show", action="store_true", help="Open browser tab after server starts.")
    args = parser.parse_args()

    def _bk_app(doc):
        modify_doc(doc)

    apps = {"/": Application(FunctionHandler(_bk_app))}

    # allow_websocket_origin can be broad for local usage
    server = Server(apps, port=args.port, allow_websocket_origin=["*"])
    server.start()

    print(f"[interactive_3d_server] Running: http://localhost:{args.port}/")
    if args.show:
        server.io_loop.add_callback(server.show, "/")

    server.io_loop.start()


if __name__ == "__main__":
    main()
else:
    # When launched via `bokeh serve`, Bokeh will create/own the doc.
    modify_doc(curdoc())

