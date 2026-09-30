"""Part 1 figures: one typical case per dataset, plus intensity / label statistics."""
import io
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import SimpleITK as sitk
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image
from skimage import measure

from common import *  # noqa: F401,F403


# ---------------------------------------------------------------- helpers
def load_canonical(path):
    img = nib.load(path)
    can = nib.as_closest_canonical(img)
    return img, can, np.asanyarray(can.dataobj)


def vox_to_canonical(orig, can, pts):
    pts = np.c_[pts, np.ones(len(pts))]
    world = pts @ orig.affine.T
    return (world @ np.linalg.inv(can.affine).T)[:, :3]


def sag(a, i):
    return a[i, ::-1, :].T


def cor(a, j):
    return a[::-1, j, :].T


def axi(a, k):
    return a[::-1, :, k].T


def overlay_contours(ax, seg2d, aspect_rows=1.0, lw=0.9):
    for lbl in np.unique(seg2d):
        if lbl == 0:
            continue
        for c in measure.find_contours((seg2d == lbl).astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=VERT_COLORS.get(int(lbl), (0.8, 0.8, 0.8)), lw=lw)


def verse_case(stem="sub-verse404_split-verse256"):
    base = DATA / "verse19"
    sub = stem.split("_")[0]
    orig, can, ct = load_canonical(base / "rawdata" / sub / f"{stem}_ct.nii.gz")
    _, _, seg = load_canonical(base / "derivatives" / sub / f"{stem}_seg-vert_msk.nii.gz")
    ctd = json.loads((base / "derivatives" / sub / f"{stem}_seg-subreg_ctd.json").read_text())[1:]
    labels = np.array([c["label"] for c in ctd])
    pts = vox_to_canonical(orig, can, np.array([[c["X"], c["Y"], c["Z"]] for c in ctd]))
    return can, ct.astype(np.float32), seg.astype(np.int16), labels, pts


# ---------------------------------------------------------------- VerSe
def fig_verse():
    can, ct, seg, labels, pts = verse_case()
    sx, sy, sz = can.header.get_zooms()[:3]
    X, Y, Z = ct.shape
    xi = int(round(np.median(pts[:, 0])))
    yj = int(round(np.median(pts[:, 1])))
    k_l1 = pts[labels == 20][0]
    zk = int(round(k_l1[2]))

    ctw = window(ct, 400, 1800)
    fig = plt.figure(figsize=(9.6, 6.2), facecolor=PLATE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 1], wspace=0.04, hspace=0.06)
    ax_s = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[:, 1])
    ax_a = fig.add_subplot(gs[0, 2])
    ax_n = fig.add_subplot(gs[1, 2])

    ax_s.imshow(sag(ctw, xi), cmap="gray", origin="lower", aspect=sz / sy)
    overlay_contours(ax_s, sag(seg, xi))
    for lbl, p in zip(labels, pts):
        ax_s.text(Y - 1 - p[1] - 34, p[2], VERT_NAMES[int(lbl)], color=VERT_COLORS[int(lbl)],
                  fontsize=6.5, ha="right", va="center", fontweight="bold")
    ax_s.set_xlim(Y - 1, 0) if False else None
    orient_marks(ax_s, "S", "I", "A", "P")
    ax_s.set_title("矢状位 · 轮廓叠加 + 质心标签", color="#DDE4E8", fontsize=9, pad=4)

    ax_c.imshow(cor(ctw, yj), cmap="gray", origin="lower", aspect=sz / sx)
    ax_c.imshow(np.ma.masked_equal(cor(seg, yj), 0), cmap=label_cmap(), vmin=0, vmax=28,
                origin="lower", aspect=sz / sx, alpha=0.45, interpolation="nearest")
    ax_c.scatter(X - 1 - pts[:, 0], pts[:, 2], s=9, c="white", edgecolors=PLATE, linewidths=0.5, zorder=3)
    orient_marks(ax_c, "S", "I", "R", "L")
    ax_c.set_title("冠状位 · 半透明填充", color="#DDE4E8", fontsize=9, pad=4)

    ax_a.imshow(axi(ctw, zk), cmap="gray", origin="lower", aspect=sy / sx)
    overlay_contours(ax_a, axi(seg, zk), lw=1.2)
    orient_marks(ax_a, "A", "P", "R", "L")
    ax_a.set_title(f"轴位 · 经 L1 质心 (z={zk})", color="#DDE4E8", fontsize=9, pad=4)

    # official-style HU profile along the spine axis, drawn inside plate
    ax_n.set_facecolor(PLATE)
    vals = ct[seg > 0]
    ax_n.hist(ct[::2, ::2, ::2].ravel(), bins=np.arange(-1100, 2000, 20), color="#6E8591", alpha=0.9,
              label="全体体素", log=True)
    ax_n.hist(vals, bins=np.arange(-1100, 2000, 20), color=VERT_COLORS[21], alpha=0.9, label="椎骨掩膜内", log=True)
    ax_n.axvspan(-500, 1300, color="white", alpha=0.06)
    ax_n.text(400, ax_n.get_ylim()[1] * 0.6, "骨窗 W1800 / L400", color="#C9D3D8", ha="center", fontsize=7.5)
    ax_n.set_xlabel("HU", color="#C9D3D8", fontsize=8)
    ax_n.tick_params(colors="#9FB0B8", labelsize=7)
    for s in ["left", "bottom"]:
        ax_n.spines[s].set_color("#4A5A62")
    ax_n.legend(frameon=False, fontsize=7, labelcolor="#DDE4E8", loc="upper right")
    ax_n.set_title("灰度分布（对数纵轴）", color="#DDE4E8", fontsize=9, pad=4)
    for ax in (ax_s, ax_c, ax_a):
        plate_axes(ax)
    scale_bar(ax_s, 1 / sy, 50)
    fig.savefig(ASSETS / "verse_ortho.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- CTSpine1K
def fig_ctspine():
    cases = [("MSD-T10", "liver_0", "liver_0_seg", "MSD T10 肝脏 CT · 层厚 5 mm"),
             ("COVID-19", "volume-covid19-A-0003_ct", "volume-covid19-A-0003_seg", "COVID-19 胸部 CT · 层厚 1 mm")]
    fig, axes = plt.subplots(1, 4, figsize=(10, 4.6), facecolor=PLATE,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1], wspace=0.03))
    for n, (src, vid, sid, title) in enumerate(cases):
        _, can, ct = load_canonical(DATA / "ctspine1k" / src / f"{vid}.nii.gz")
        _, _, seg = load_canonical(DATA / "ctspine1k" / src / f"{sid}.nii.gz")
        sx, sy, sz = can.header.get_zooms()[:3]
        xs, ys, zs = np.nonzero(seg)
        xi = int(np.median(xs))
        yj = int(np.median(ys))
        ctw = window(ct.astype(np.float32), 400, 1800)
        a0, a1 = axes[2 * n], axes[2 * n + 1]
        a0.imshow(sag(ctw, xi), cmap="gray", origin="lower", aspect=sz / sy, interpolation="nearest")
        a1.imshow(sag(ctw, xi), cmap="gray", origin="lower", aspect=sz / sy, interpolation="nearest")
        a1.imshow(np.ma.masked_equal(sag(seg, xi), 0), cmap=label_cmap(), vmin=0, vmax=28, origin="lower",
                  aspect=sz / sy, alpha=0.55, interpolation="nearest")
        for a in (a0, a1):
            plate_axes(a)
            zlo, zhi = zs.min() - 30 / sz, zs.max() + 30 / sz
            a.set_ylim(max(zlo, 0), min(zhi, ct.shape[2] - 1))
        a0.set_title(title, color="#DDE4E8", fontsize=8.5, loc="left", pad=4)
        present = sorted(int(l) for l in np.unique(seg) if l)
        a1.set_title(f"标签 {VERT_NAMES[present[0]]}–{VERT_NAMES[present[-1]]}", color="#DDE4E8",
                     fontsize=8.5, loc="left", pad=4)
        scale_bar(a0, 1 / sy, 50)
    fig.savefig(ASSETS / "ctspine_compare.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- SPIDER
SPIDER_VERT = ["#F2CF7E", "#E8B868", "#DDA257", "#D18C45", "#C0782B", "#A9651F", "#8F5418", "#7A4613", "#653A10"]


def spider_color(lbl):
    if lbl == 100:
        return "#58B8A5"
    if lbl > 200:
        return "#7FB2E0"
    return SPIDER_VERT[(lbl - 1) % len(SPIDER_VERT)]


def fig_spider():
    items = [("1_t1", "病例 1 · T1"), ("1_t2", "病例 1 · T2"), ("100_t2", "病例 100 · T2")]
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.55), facecolor=PLATE, gridspec_kw=dict(wspace=0.03))
    for n, (name, title) in enumerate(items):
        im = sitk.ReadImage(str(DATA / "spider" / "images" / f"{name}.mha"))
        mk = sitk.ReadImage(str(DATA / "spider" / "masks" / f"{name}.mha"))
        a = sitk.GetArrayFromImage(im).astype(np.float32)  # z, y, x
        m = sitk.GetArrayFromImage(mk).astype(np.int16)
        sx, sy, sz = im.GetSpacing()
        xi = int(np.median(np.nonzero(m == 100)[2])) if (m == 100).any() else a.shape[2] // 2
        sl = a[:, :, xi]
        ml = m[:, :, xi]
        body = sl[(sl > -1000) & (sl < 3096)]
        lo, hi = np.percentile(body, [1, 99])
        a0, a1 = axes[2 * n], axes[2 * n + 1]
        for ax in (a0, a1):
            ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=sz / sy)
            plate_axes(ax)
        for lbl in np.unique(ml):
            if lbl == 0:
                continue
            for c in measure.find_contours((ml == lbl).astype(float), 0.5):
                a1.plot(c[:, 1], c[:, 0], color=spider_color(int(lbl)), lw=0.9)
        a0.set_title(title, color="#DDE4E8", fontsize=8.5, loc="left", pad=4)
        a1.set_title(f"{sl.shape[1]}×{sl.shape[0]}×{a.shape[2]}\n层厚 {sx:.1f} mm", color="#9FB0B8",
                     fontsize=7.5, loc="left", pad=4)
        scale_bar(a0, 1 / sy, 50)
    fig.savefig(ASSETS / "spider_t1t2.png", facecolor=PLATE)
    plt.close(fig)

    g = pd.read_csv(DATA / "spider" / "radiological_gradings.csv")
    tab = pd.crosstab(g["IVD label"], g["Pfirrman grade"])
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=8,
                    color="white" if v > tab.values.max() * 0.55 else INK)
    ax.set_xticks(range(tab.shape[1]), [f"{c}" for c in tab.columns])
    ax.set_yticks(range(tab.shape[0]), [f"IVD {r}" for r in tab.index])
    ax.set_xlabel("Pfirrmann 分级（1 = 正常，5 = 重度退变）")
    ax.set_title(f"SPIDER 椎间盘退变分级  n = {len(g)} 个椎间盘", loc="left")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("椎间盘数", fontsize=8)
    fig.savefig(ASSETS / "spider_grades.svg")
    plt.close(fig)


# ---------------------------------------------------------------- RSNA 2024
COND_ZH = {
    "spinal_canal_stenosis": "椎管狭窄",
    "left_neural_foraminal_narrowing": "左侧椎间孔狭窄",
    "right_neural_foraminal_narrowing": "右侧椎间孔狭窄",
    "left_subarticular_stenosis": "左侧侧隐窝狭窄",
    "right_subarticular_stenosis": "右侧侧隐窝狭窄",
}
SEV = ["Normal/Mild", "Moderate", "Severe"]
SEV_COL = ["#9DB7C7", "#25577A", BONE]


def fig_rsna():
    t = pq.read_table(DATA / "rsna2024" / "train.parquet").to_pandas()
    meta = t["metadata"].map(json.loads)
    t["level"] = meta.map(lambda d: d["level"])
    t["ac"] = pd.read_pickle(DATA / "rsna2024" / "ac.pkl")["ac"].values
    rng = np.random.default_rng(7)
    conds = list(COND_ZH)
    fig, axes = plt.subplots(len(conds), 6, figsize=(8.4, 7.4), facecolor=PLATE,
                             gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for r, c in enumerate(conds):
        for s in range(3):
            idx = t.index[(t.condition == c) & (t.label == s) & (t.ac >= 0.85)]
            pick = rng.choice(idx, 2, replace=False)
            for k, i in enumerate(pick):
                ax = axes[r, s * 2 + k]
                im = Image.open(io.BytesIO(t.at[i, "image"]["bytes"]))
                ax.imshow(np.asarray(im), cmap="gray")
                plate_axes(ax)
                ax.text(0.03, 0.04, t.at[i, "level"].upper().replace("_", "/"), color="#DDE4E8",
                        fontsize=6.5, transform=ax.transAxes)
                if r == 0 and k == 0:
                    ax.set_title(SEV[s], color=SEV_COL[s] if s else "#DDE4E8", fontsize=9, loc="left")
        axes[r, 0].set_ylabel(COND_ZH[c], color="#DDE4E8", fontsize=8.5, rotation=0, ha="right", va="center")
        axes[r, 0].yaxis.set_label_coords(-0.06, 0.5)
    fig.savefig(ASSETS / "rsna_grid.png", facecolor=PLATE)
    plt.close(fig)

    ct = pd.crosstab(t.condition, t.label).reindex(conds)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    left = np.zeros(len(conds))
    for s in range(3):
        v = ct[s].values
        ax.barh(range(len(conds)), v, left=left, color=SEV_COL[s], label=SEV[s], height=0.62)
        for i, (l, w) in enumerate(zip(left, v)):
            if w > 700:
                ax.text(l + w / 2, i, f"{w/ct.values[i].sum():.0%}", ha="center", va="center",
                        fontsize=7.5, color="white" if s else INK)
        left += v
    ax.set_yticks(range(len(conds)), [COND_ZH[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel("图像数（HF 镜像，n = 30,022）")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(ASSETS / "rsna_dist.svg")
    plt.close(fig)
    return t


# ---------------------------------------------------------------- CSXA
def csxa_table():
    x = pd.read_excel(DATA / "csxa" / "datasets.xlsx", header=None)
    body = x.iloc[2:].copy()
    df = pd.DataFrame({
        "number": pd.to_numeric(body[0], errors="coerce"),
        "female": pd.to_numeric(body[1], errors="coerce"),
        "age": pd.to_numeric(body[2], errors="coerce"),
        "disease": pd.to_numeric(body[3], errors="coerce"),
        "curv": pd.to_numeric(body[4], errors="coerce"),
        "cobb27": pd.to_numeric(body[106], errors="coerce"),
        "sva": pd.to_numeric(body[108], errors="coerce"),
    }).dropna(subset=["number"])
    return df


CURV = {1: "前凸", 2: "变直", 3: "S 形 1", 4: "S 形 2", 5: "后凸"}


def fig_csxa():
    df = csxa_table()
    pngs = sorted((DATA / "csxa" / "datasets-PNG").glob("*.png"))
    pick = pngs[:4]
    fig, axes = plt.subplots(1, len(pick), figsize=(10, 4.4), facecolor=PLATE, gridspec_kw=dict(wspace=0.03))
    for ax, p in zip(axes, pick):
        im = np.asarray(Image.open(p).convert("L"))
        ax.imshow(im, cmap="gray")
        plate_axes(ax)
        j = json.loads((DATA / "csxa" / "datasets-JSON" / f"{p.stem}.json").read_text(encoding="utf-8"))
        P = {s["label"].strip(): np.array(s["points"][0]) for s in j["shapes"]}
        for v in ["C3", "C4", "C5", "C6", "C7"]:
            q = [P.get(f"{v} top left"), P.get(f"{v} top right"), P.get(f"{v} bottom right"), P.get(f"{v} bottom left")]
            if all(e is not None for e in q):
                ax.add_patch(Polygon(np.array(q), closed=True, fill=False, ec=VERT_COLORS[int(v[1])], lw=1.1))
        for a, b in [("C2 bottom left", "C2 bottom right"), ("C7 bottom left", "C7 bottom right")]:
            if a in P and b in P:
                d = P[b] - P[a]
                d = d / np.linalg.norm(d)
                ax.plot(*np.c_[P[a] - d * 60, P[b] + d * 60], color=BONE, lw=1.2, ls="--")
        pts = np.array(list(P.values()))
        ax.scatter(pts[:, 0], pts[:, 1], s=5, c="white", zorder=3)
        y0, y1 = pts[:, 1].min() - 50, pts[:, 1].max() + 50
        cx, hw = pts[:, 0].mean(), (y1 - y0) * 0.55
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(y1, y0)
        num, sexd, age = int(p.stem[:4]), int(p.stem[4]), int(p.stem[5:])
        row = df[df.number == num]
        cobb = row.cobb27.iloc[0] if len(row) else np.nan
        curv = CURV.get(int(row.curv.iloc[0]), "?") if len(row) else "?"
        ax.set_title(f"#{num} · {'女' if sexd else '男'} {age} 岁 · {curv}\n"
                     f"Cobb {cobb:.1f}° · {im.shape[1]}×{im.shape[0]} px",
                     color="#DDE4E8", fontsize=8, loc="left", pad=4, linespacing=1.5)
    fig.savefig(ASSETS / "csxa_xray.png", facecolor=PLATE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    data = [df.loc[df.curv == k, "cobb27"].dropna().values for k in CURV]
    parts = ax.violinplot(data, positions=range(5), showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor("#9DB7C7")
        b.set_alpha(0.55)
        b.set_edgecolor("none")
    ax.boxplot(data, positions=range(5), widths=0.18, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", color=FILM), medianprops=dict(color=BONE, lw=2),
               whiskerprops=dict(color=FILM), capprops=dict(color=FILM))
    ax.axhline(0, color=SLATE, lw=0.6, ls=":")
    ax.set_xticks(range(5), [f"{CURV[k]}\nn={len(d)}" for k, d in zip(CURV, data)])
    ax.set_ylabel("C2–7 Cobb 角 (°)")
    ax.set_title("不同曲度分型的 Cobb 角分布", loc="left")
    fig.savefig(ASSETS / "csxa_cobb.svg")
    plt.close(fig)
    return df


# ---------------------------------------------------------------- intensity comparison
def fig_intensity(rsna_t):
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.5), gridspec_kw=dict(wspace=0.35))
    _, ct, _, _, _ = verse_case()
    ax = axes[0]
    ax.hist(ct[::2, ::2, ::2].ravel(), bins=np.arange(-1100, 2200, 25), color=FILM, log=True)
    ax.set_title("CT (VerSe) · HU，物理量", loc="left", fontsize=9)
    ax.set_xlabel("HU")
    for x, t in [(-1000, "空气"), (0, "水"), (700, "皮质骨")]:
        ax.axvline(x, color=BONE, lw=0.8, ls="--")
        ax.text(x + 40, 0.97, t, fontsize=7, color=BONE, ha="left", va="top",
                transform=ax.get_xaxis_transform())

    im = sitk.GetArrayFromImage(sitk.ReadImage(str(DATA / "spider" / "images" / "1_t1.mha"))).ravel()
    im2 = sitk.GetArrayFromImage(sitk.ReadImage(str(DATA / "spider" / "images" / "1_t2.mha"))).ravel()
    ax = axes[1]
    b = np.arange(-1000, 3150, 40)
    ax.hist(im, bins=b, color=FILM, alpha=0.75, log=True, label="T1")
    ax.hist(im2, bins=b, color=BONE, alpha=0.6, log=True, label="T2")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("MRI (SPIDER) · 无物理单位", loc="left", fontsize=9)
    ax.set_xlabel("存储值 (int16)")

    ax = axes[2]
    vals = np.concatenate([np.asarray(Image.open(p).convert("L")).ravel()
                           for p in sorted((DATA / "csxa" / "datasets-PNG").glob("*.png"))[:4]])
    ax.hist(vals, bins=np.arange(0, 257, 4), color=FILM, log=True)
    ax.set_title("X 光 (CSXA) · 8-bit PNG", loc="left", fontsize=9)
    ax.set_xlabel("灰度 0–255")

    ax = axes[3]
    sub = rsna_t.sample(300, random_state=0)
    v = np.concatenate([np.asarray(Image.open(io.BytesIO(r["bytes"]))).ravel() for r in sub["image"]])
    ax.hist(v, bins=np.arange(0, 257, 4), color=FILM, log=True)
    ax.set_title("MRI (RSNA 镜像) · 已归一化", loc="left", fontsize=9)
    ax.set_xlabel("灰度 0–255")
    for a in axes:
        a.tick_params(labelsize=7)
    fig.savefig(ASSETS / "intensity_compare.svg")
    plt.close(fig)


if __name__ == "__main__":
    fig_verse()
    print("verse ok")
    fig_ctspine()
    print("ctspine ok")
    fig_spider()
    print("spider ok")
    t = fig_rsna()
    print("rsna ok")
    df = fig_csxa()
    print("csxa ok", df.shape, df.curv.value_counts().to_dict(), df.cobb27.describe().round(1).to_dict())
    fig_intensity(t)
    print("intensity ok")
