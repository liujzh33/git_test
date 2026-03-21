from __future__ import annotations

import argparse
import math
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import triangulate, unary_union

from generator import GenerationConfig, generate_red_and_green
from grid_assign import build_clipped_grid
from grid_assign import assign_properties_to_cells


MINERALS = ["FeO", "TiO", "Al2O3", "SiO2"]
MINERAL_COLORS = {
    "FeO": "#d73027",
    "TiO": "#4575b4",
    "Al2O3": "#1a9850",
    "SiO2": "#984ea3",
}


def _dominant_mineral(row: Dict[str, float]) -> str:
    best = "FeO"
    best_v = -1.0
    for m in MINERALS:
        v = float(row.get(m, 0.0) or 0.0)
        if v > best_v:
            best_v = v
            best = m
    return best


def _total_mineral(row: Dict[str, float]) -> float:
    return sum(float(row.get(m, 0.0) or 0.0) for m in MINERALS)


def _triangles_from_polygon(poly: Polygon, simplify_tol: float) -> List[Polygon]:
    """
    Triangulate a polygon footprint and keep triangles whose centroid lies inside.
    """
    if poly.is_empty or poly.area <= 1e-9:
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
        # centroid-in-polygon is robust for filtering outside triangles
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
    z_value: float | None,
    z_func=None,
    face_color: str = "#999999",
) -> Tuple[go.Mesh3d, int]:
    """
    Build a Plotly Mesh3d where each triangle has independent vertices (no sharing),
    making i/j/k indexing trivial.
    """
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
        # triangle exterior ends with repeated first point
        (x0, y0), (x1, y1), (x2, y2) = coords[0], coords[1], coords[2]
        z = float(z_value) if z_value is not None else float(z_func(tri))

        base_idx = len(xs)
        xs.extend([x0, x1, x2])
        ys.extend([y0, y1, y2])
        zs.extend([z, z, z])

        i.append(base_idx)
        j.append(base_idx + 1)
        k.append(base_idx + 2)
        tri_count += 1

    mesh = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        color=face_color,
        opacity=0.95,
        flatshading=True,
        name="mesh",
    )
    return mesh, tri_count


def export_3d_html(
    cell_size: int,
    green_zone_count: int,
    out_html: str,
    seed: int = 77,
    base_amp: float = 20.0,
    top_amp: float = 60.0,
    triang_simplify_ratio_red: float = 0.02,
    triang_simplify_ratio_black: float = 0.01,
) -> None:
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

    red_gdf, green_gdf, bbox_geom = generate_red_and_green(cfg)
    green_union = unary_union(green_gdf.geometry.tolist()).buffer(0)

    black_grid = build_clipped_grid(green_geom=green_union, bbox_geom=bbox_geom, cell_size=cfg.cell_size)
    black_gdf = assign_properties_to_cells(black_grid, red_gdf)

    bbox = box(cfg.x_min, cfg.y_min, cfg.x_max, cfg.y_max)

    # base height per red polygon
    red_records = red_gdf.to_dict("records")
    color_vals = [float(r.get("color_value", 0.0) or 0.0) for r in red_records]
    cmin, cmax = (min(color_vals), max(color_vals)) if color_vals else (0.0, 1.0)
    denom = (cmax - cmin) if (cmax - cmin) != 0 else 1.0

    z_base_by_red: Dict[str, float] = {}
    dom_mineral_by_red: Dict[str, str] = {}
    for r in red_records:
        rid = str(r["red_id"])
        cv = float(r.get("color_value", 0.0) or 0.0)
        t = (cv - cmin) / denom
        z_base_by_red[rid] = base_amp * (0.25 + 0.75 * t)
        mineral_row = {m: float(r.get(m, 0.0) or 0.0) for m in MINERALS}
        dom_mineral_by_red[rid] = _dominant_mineral(mineral_row)

    # uplift normalization for black cells
    black_records = black_gdf.to_dict("records")
    totals = [_total_mineral(r) for r in black_records]
    tot_min, tot_max = (min(totals), max(totals)) if totals else (0.0, 1.0)
    tot_denom = (tot_max - tot_min) if (tot_max - tot_min) != 0 else 1.0

    # ---- build 3D meshes ----
    tri_simplify_red = cfg.cell_size * triang_simplify_ratio_red
    tri_simplify_black = cfg.cell_size * triang_simplify_ratio_black

    fig = go.Figure()

    # Base: build per-red polygon triangles (but using constant z per red polygon).
    base_tri_total = 0
    for r in red_records:
        rid = str(r["red_id"])
        dom = dom_mineral_by_red.get(rid, "FeO")
        color = MINERAL_COLORS.get(dom, "#999999")
        z_base = z_base_by_red.get(rid, 0.0)
        tris = _triangles_from_geom(r["geometry"], tri_simplify_red)
        mesh, tri_count = _mesh_from_triangles(tris, z_value=z_base, face_color=color)
        mesh.name = f"base_{rid}"
        base_tri_total += tri_count
        fig.add_trace(mesh)

    # Top: split by dominant mineral to get stable coloring.
    top_tris_by_mineral: Dict[str, List[Polygon]] = {m: [] for m in MINERALS}
    top_z_by_cell: Dict[str, float] = {}
    top_dom_by_cell: Dict[str, str] = {}

    for rec in black_records:
        cell_id = str(rec["cell_id"])
        red_id = str(rec["red_id"])
        z_base = z_base_by_red.get(red_id, 0.0)
        ttot = (_total_mineral(rec) - tot_min) / tot_denom
        z_top = z_base + top_amp * ttot
        top_z_by_cell[cell_id] = z_top

        dom = _dominant_mineral(rec)
        top_dom_by_cell[cell_id] = dom

        tris = _triangles_from_geom(rec["geometry"], tri_simplify_black)
        # store triangles into mineral group; z will be looked up using closure index (cell_id)
        # Here we cannot associate cell_id with each triangle easily without object wrapper,
        # so we build cells as independent Meshes below instead.
        # To keep simpler, we do cell-by-cell trace aggregation.
        for m in []:  # no-op
            pass

    # Build top meshes cell-by-cell but merge by mineral trace count.
    # For simplicity we create up to 4 traces (one per mineral), using independent vertices per triangle.
    # We'll collect triangles as (tri, z, color) per mineral.
    top_geom_by_mineral: Dict[str, List[Tuple[Polygon, float, str]]] = {m: [] for m in MINERALS}
    for rec in black_records:
        cell_id = str(rec["cell_id"])
        dom = top_dom_by_cell.get(cell_id, "FeO")
        z_top = top_z_by_cell.get(cell_id, 0.0)
        color = MINERAL_COLORS.get(dom, "#cccccc")
        tris = _triangles_from_geom(rec["geometry"], tri_simplify_black)
        for tri in tris:
            top_geom_by_mineral[dom].append((tri, z_top, color))

    top_tri_total = 0
    for dom_m, tri_list in top_geom_by_mineral.items():
        if not tri_list:
            continue
        # Build mesh from triangles with per-triangle constant z (already stored in tri_list).
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        i: List[int] = []
        j: List[int] = []
        k: List[int] = []
        tri_count = 0
        for tri, z_top, color in tri_list:
            coords = list(tri.exterior.coords)
            if len(coords) < 4:
                continue
            (x0, y0), (x1, y1), (x2, y2) = coords[0], coords[1], coords[2]
            base_idx = len(xs)
            xs.extend([x0, x1, x2])
            ys.extend([y0, y1, y2])
            zs.extend([z_top, z_top, z_top])
            i.append(base_idx)
            j.append(base_idx + 1)
            k.append(base_idx + 2)
            tri_count += 1
        top_tri_total += tri_count
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i,
                j=j,
                k=k,
                color=MINERAL_COLORS.get(dom_m, "#cccccc"),
                opacity=0.92,
                flatshading=True,
                name=f"top_{dom_m}",
            )
        )

    # ---- camera & styling ----
    fig.update_layout(
        title=f"3D Mountain Mesh (cell_size={cell_size}m, green_zones={green_zone_count})",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            aspectmode="data",
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    # A bit of nicer default camera:
    fig.update_layout(scene_camera=dict(eye=dict(x=1.35, y=1.35, z=0.7)))

    fig.write_html(out_html, include_plotlyjs="cdn")

    print(f"[3d] base triangles: {base_tri_total}")
    print(f"[3d] top triangles: {top_tri_total}")
    print(f"[3d] saved: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 3D mountain mesh from generated 2D blocks.")
    parser.add_argument("--green_zone_count", type=int, default=3)
    parser.add_argument("--cell_size", type=int, default=300)
    parser.add_argument("--out", type=str, default="output/3d_mountain.html")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--base_amp", type=float, default=20.0, help="Base elevation amplitude")
    parser.add_argument("--top_amp", type=float, default=60.0, help="Top uplift amplitude")
    args = parser.parse_args()

    export_3d_html(
        cell_size=args.cell_size,
        green_zone_count=args.green_zone_count,
        out_html=args.out,
        seed=args.seed,
        base_amp=args.base_amp,
        top_amp=args.top_amp,
    )


if __name__ == "__main__":
    main()

