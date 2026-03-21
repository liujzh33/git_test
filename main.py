from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.ops import unary_union

from generator import GenerationConfig, generate_red_and_green
from grid_assign import assign_properties_to_cells, build_clipped_grid, export_outputs
from validate import validate_all
from interactive_generate import generate_interactive_html


def plot_result(red_gdf, green_gdf, black_gdf, bbox_geom, cell_size: int, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=170)

    red_gdf.plot(ax=ax, column="color_value", cmap="Reds", edgecolor="#b00020", linewidth=1.3, alpha=0.62, legend=True)
    black_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.35, alpha=0.90)
    for i, g in enumerate(green_gdf.geometry.tolist()):
        xg, yg = g.exterior.xy
        label = "Green Zones" if i == 0 else None
        ax.plot(xg, yg, color="#2bbf4b", linewidth=2.0, label=label)

    xb, yb = bbox_geom.exterior.xy
    ax.plot(xb, yb, color="#2f2f2f", linewidth=1.5, linestyle="--", label="Bounding Box")

    ax.set_title(f"Constrained Mining Blocks + {cell_size}m Clipped Grid", fontsize=14, pad=12)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(alpha=0.18, linestyle=":")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mining blocks demo + interactive HTML.")
    parser.add_argument("--red_count_min", type=int, default=28)
    parser.add_argument("--red_count_max", type=int, default=34)
    parser.add_argument("--min_shared_edge_m", type=int, default=90)
    parser.add_argument("--green_area_ratio_min", type=float, default=0.30)
    parser.add_argument("--green_area_ratio_max", type=float, default=0.60)
    parser.add_argument("--green_zone_count_min", type=int, default=3)
    parser.add_argument("--green_zone_count_max", type=int, default=3)
    parser.add_argument("--cell_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--minerals_per_red_min", type=int, default=1)
    parser.add_argument("--minerals_per_red_max", type=int, default=3)
    parser.add_argument("--no_interactive", action="store_true")
    args = parser.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerationConfig(
        red_count_min=args.red_count_min,
        red_count_max=args.red_count_max,
        min_shared_edge_m=args.min_shared_edge_m,
        green_area_ratio_min=args.green_area_ratio_min,
        green_area_ratio_max=args.green_area_ratio_max,
        green_zone_count_min=args.green_zone_count_min,
        green_zone_count_max=args.green_zone_count_max,
        cell_size=args.cell_size,
        seed=args.seed,
        minerals_per_red_min=args.minerals_per_red_min,
        minerals_per_red_max=args.minerals_per_red_max,
    )

    red_gdf, green_gdf, bbox_geom = generate_red_and_green(cfg)
    green_union = unary_union(green_gdf.geometry.tolist()).buffer(0)
    black_grid = build_clipped_grid(green_geom=green_union, bbox_geom=bbox_geom, cell_size=cfg.cell_size)
    result = assign_properties_to_cells(black_grid, red_gdf)

    out_geojson = out_dir / "grid_with_properties.geojson"
    out_csv = out_dir / "grid_with_properties.csv"
    out_png = out_dir / "result.png"
    export_outputs(result, str(out_geojson), str(out_csv))
    plot_result(red_gdf, green_gdf, result, bbox_geom, cfg.cell_size, out_png)

    report = validate_all(red_gdf, green_gdf, result)
    print("==== Validation Report ====")
    print(f"Red block count: {report.red_count}")
    print(f"Red overlap pairs (must be 0): {report.overlap_pairs}")
    print(f"Minimum shared boundary among adjacent reds (m): {report.min_max_shared_edge:.2f}")
    print(f"Green zone count: {report.green_count}")
    print(f"Green within red union: {report.green_within_red_union}")
    print(f"Black cells kept: {report.black_cells}")
    print(f"Clipped cell ratio: {report.clipped_ratio:.2%}")
    print(f"Unassigned black cells (must be 0): {report.unassigned_cells}")
    print(f"Saved: {out_geojson}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")

    if not args.no_interactive:
        out_html = out_dir / "interactive.html"
        generate_interactive_html(cfg, str(out_html), red_gdf=red_gdf, green_gdf=green_gdf, bbox_geom=bbox_geom, black_grid=black_grid)


if __name__ == "__main__":
    main()
