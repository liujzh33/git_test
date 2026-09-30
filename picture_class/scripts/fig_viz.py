"""Part 3 figures: ways to put a 3D spine volume onto a 2D slide."""
import json

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from skimage import measure

from common import *  # noqa: F401,F403
from fig_datasets import axi, cor, overlay_contours, sag, verse_case

can, CT, SEG, LABELS, PTS = verse_case()
SP = np.array(can.header.get_zooms()[:3], dtype=float)
order = np.argsort(LABELS)
LABELS, PTS = LABELS[order], PTS[order]
CTW = window(CT, 400, 1800)
TEXT = "#DDE4E8"


# ---------------------------------------------------------------- spine frame (for CPR / oblique cuts)
def spine_frame(step=1.0, extend=25.0):
    p = PTS * SP
    tck, _ = splprep(p.T, s=len(p) * 6.0, k=3)
    u = np.linspace(0, 1, 4000)
    c = np.array(splev(u, tck)).T
    seg_len = np.linalg.norm(np.diff(c, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg_len)]
    s = np.arange(-extend, arc[-1] + extend, step)
    uu = np.interp(np.clip(s, 0, arc[-1]), arc, u)
    cen = np.array(splev(uu, tck)).T
    tan = np.array(splev(uu, tck, der=1)).T
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    lo, hi = s < 0, s > arc[-1]
    cen[lo] = cen[~lo][0] + tan[~lo][0] * s[lo, None]
    cen[hi] = cen[~hi][-1] + tan[~hi][-1] * (s[hi] - arc[-1])[:, None]
    ex = np.array([1.0, 0, 0])
    n1 = ex - (tan @ ex)[:, None] * tan
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(tan, n1)
    if np.mean(n2[:, 1]) > 0:  # make +n2 point posterior
        n2 = -n2
    arc_ctd = np.interp(np.linspace(0, 1, len(p)), u, arc)
    # arc position of each centroid: nearest curve sample
    arc_ctd = np.array([s[np.argmin(np.linalg.norm(cen - q, axis=1))] for q in p])
    return s, cen, tan, n1, n2, arc_ctd


def sample(vol, pts_mm, order=1):
    vox = (pts_mm / SP).reshape(-1, 3).T
    out = map_coordinates(vol, vox, order=order, mode="constant", cval=vol.min() if order else 0)
    return out.reshape(pts_mm.shape[:-1])


S, CEN, TAN, N1, N2, ARC_CTD = spine_frame()


def cpr(direction, half=70.0, step=0.8):
    off = np.arange(-half, half + step, step)
    vec = N2 if direction == "sag" else N1
    grid = CEN[:, None, :] + off[None, :, None] * vec[:, None, :]
    img = sample(CTW, grid)
    lab = sample(SEG, grid, order=0).astype(int)
    if direction == "cor":
        img, lab = img[:, ::-1], lab[:, ::-1]
    return img, lab, step


def oblique(k, half=48.0, step=0.5, ap=(-38.0, 62.0)):
    i = np.argmin(np.abs(S - ARC_CTD[k]))
    a = np.arange(-half, half + step, step)
    r = np.arange(ap[0], ap[1] + step, step)
    rows, cols = np.meshgrid(r, -a, indexing="ij")  # rows: anterior->posterior; cols: R->L
    grid = CEN[i] + rows[..., None] * N2[i] + cols[..., None] * N1[i]
    return sample(CTW, grid), sample(SEG, grid, order=0).astype(int)


# ---------------------------------------------------------------- 1. windowing
def fig_windows():
    k = int(round(PTS[LABELS == 20][0][2]))
    specs = [("骨窗 W1800 / L400", 400, 1800), ("软组织窗 W400 / L40", 40, 400), ("全范围线性映射", 500, 3200)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor=PLATE, gridspec_kw=dict(wspace=0.03))
    for ax, (t, l, w) in zip(axes, specs):
        sl = axi(window(CT, l, w), k)
        ax.imshow(sl, cmap="gray", origin="lower", aspect=SP[1] / SP[0], interpolation="bilinear")
        rows = np.nonzero(np.median(axi(CT, k), axis=1) > -700)[0]
        ax.set_ylim(rows.min() - 0.5, rows.max() + 0.5)
        plate_axes(ax)
        ax.set_title(t, color=TEXT, fontsize=9, loc="left")
    fig.savefig(ASSETS / "viz_windows.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- 2. projections
def fig_projections():
    yj = int(round(np.median(PTS[:, 1])))
    single = cor(CTW, yj)
    body = CT.copy()
    mip = window(body.max(axis=1), 600, 1800)[::-1].T
    mean = body.clip(-1000, None).mean(axis=1)
    mean = ((mean - np.percentile(mean, 1)) / (np.percentile(mean, 99.8) - np.percentile(mean, 1))).clip(0, 1)[::-1].T
    lab = SEG.max(axis=1)[::-1].T
    tiles = [(single, None, "单张冠状切片"), (mip, None, "最大密度投影 MIP"),
             (mean, None, "平均投影（近似 X 光 / DRR）"), (mean, lab, "投影 + 标签")]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 6.0), facecolor=PLATE, gridspec_kw=dict(wspace=0.03))
    for ax, (im, lb, t) in zip(axes, tiles):
        ax.imshow(im, cmap="gray", origin="lower", aspect=SP[2] / SP[0])
        if lb is not None:
            ax.imshow(np.ma.masked_equal(lb, 0), cmap=label_cmap(), vmin=0, vmax=28, alpha=0.55,
                      origin="lower", aspect=SP[2] / SP[0], interpolation="nearest")
        plate_axes(ax)
        ax.set_title(t, color=TEXT, fontsize=9, loc="left")
    scale_bar(axes[0], 1 / SP[0], 50)
    fig.savefig(ASSETS / "viz_projection.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- 3. lightbox, one tile per vertebra
def fig_lightbox():
    n = len(LABELS)
    cols = 9
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.12 * rows), facecolor=PLATE,
                             gridspec_kw=dict(wspace=0.03, hspace=0.04))
    for ax in axes.ravel():
        plate_axes(ax)
    for k, (lbl, ax) in enumerate(zip(LABELS, axes.ravel())):
        img, lab = oblique(k)
        ax.imshow(img, cmap="gray", origin="upper")
        for c in measure.find_contours((lab == lbl).astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=VERT_COLORS[int(lbl)], lw=1)
        ax.text(0.05, 0.93, VERT_NAMES[int(lbl)], color=VERT_COLORS[int(lbl)], fontsize=9, fontweight="bold",
                transform=ax.transAxes, va="top")
    axes.ravel()[0].text(0.95, 0.07, "A↑ R←", color="#9FB0B8", fontsize=6.5, ha="right", transform=axes.ravel()[0].transAxes)
    fig.savefig(ASSETS / "viz_lightbox.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- 4. curved planar reformation
def fig_cpr():
    xi = int(round(np.median(PTS[:, 0])))
    sag_img = sag(CTW, xi)
    s_img, s_lab, st = cpr("sag")
    c_img, c_lab, _ = cpr("cor")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 6.6), facecolor=PLATE,
                             gridspec_kw=dict(wspace=0.05, width_ratios=[sag_img.shape[1] * SP[1], 140, 140]))
    axes[0].imshow(sag_img, cmap="gray", origin="lower", aspect=SP[2] / SP[1])
    axes[0].plot(SEG.shape[1] - 1 - CEN[:, 1] / SP[1], CEN[:, 2] / SP[2], color=BONE, lw=1.2)
    axes[0].set_title("原始矢状切片\n黄线 = 拟合的椎体中心线", color=TEXT, fontsize=8.5, loc="left")
    for ax, im, lb, t in [(axes[1], s_img, s_lab, "CPR 拉直 · 矢状"), (axes[2], c_img, c_lab, "CPR 拉直 · 冠状")]:
        ax.imshow(im, cmap="gray", origin="upper", aspect=1)
        overlay_contours(ax, lb, lw=0.8)
        ax.set_title(t, color=TEXT, fontsize=8.5, loc="left")
    for k, lbl in enumerate(LABELS):
        r = (ARC_CTD[k] - S[0]) / (S[1] - S[0])
        axes[2].text(c_img.shape[1] + 3, r, VERT_NAMES[int(lbl)], color=VERT_COLORS[int(lbl)],
                     fontsize=6.5, va="center", clip_on=False)
    for ax in axes:
        plate_axes(ax)
    orient_marks(axes[1], "S", "I", "A", "P")
    orient_marks(axes[2], "S", "I", "R", "L")
    scale_bar(axes[1], 1 / st, 50)
    fig.savefig(ASSETS / "viz_cpr.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- 5. 3D surface render
def surface_mesh(step=2):
    verts_all, faces_all, cols = [], [], []
    off = 0
    for lbl in LABELS:
        m = SEG == lbl
        if m.sum() < 50:
            continue
        idx = np.argwhere(m)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = idx.max(0) + 3
        sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        sub = gaussian_filter(sub, sigma=0.9 / SP * SP.min())
        v, f, n, _ = measure.marching_cubes(sub, 0.5, spacing=tuple(SP), step_size=step)
        v += lo * SP
        verts_all.append(v)
        faces_all.append(f + off)
        cols += [VERT_COLORS[int(lbl)]] * len(f)
        off += len(v)
    return np.vstack(verts_all), np.vstack(faces_all), np.array(cols)


def shade(tri, base, light):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    lam = np.abs(n @ light)
    return np.clip(base * (0.30 + 0.75 * lam[:, None]), 0, 1)


def render(ax, verts, faces, cols, elev, azim, light):
    tri = verts[faces]
    fc = shade(tri, cols, light / np.linalg.norm(light))
    pc = Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    mn, mx = verts.min(0), verts.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_box_aspect(mx - mn)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor(PLATE)


def render_view(verts, faces, cols, elev, azim, light, bg):
    fig = plt.figure(figsize=(6, 12), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    render(ax, verts, faces, cols, elev, azim, light)
    ax.set_facecolor(bg)
    ax.set_box_aspect(verts.max(0) - verts.min(0), zoom=1.0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    bgc = (np.array(to_rgb(bg)) * 255).astype(int)
    fg = np.abs(img.astype(int) - bgc).sum(-1) > 12
    r, c = np.nonzero(fg)
    pad = 12
    return img[max(r.min() - pad, 0):r.max() + pad, max(c.min() - pad, 0):c.max() + pad]


VIEWS = [(6, 180, np.array([-1, 0.5, 0.4]), "左侧观"), (6, 90, np.array([-0.4, 1, 0.4]), "前面观"),
         (16, 135, np.array([-1, 1, 0.7]), "左前斜")]


def fig_render3d():
    verts, faces, cols = surface_mesh()
    crops = [render_view(verts, faces, cols, e, a, L, PLATE) for e, a, L, _ in VIEWS]
    h = max(c.shape[0] for c in crops)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.2), facecolor=PLATE,
                             gridspec_kw=dict(wspace=0.02, width_ratios=[c.shape[1] for c in crops]))
    for ax, c, (_, _, _, t) in zip(axes, crops, VIEWS):
        canvas = np.zeros((h, c.shape[1], 3), np.uint8) + (np.array(to_rgb(PLATE)) * 255).astype(np.uint8)
        canvas[h - c.shape[0]:] = c
        ax.imshow(canvas)
        plate_axes(ax)
        ax.set_title(t, color=TEXT, fontsize=9, loc="left")
    handles = [Patch(color=VERT_COLORS[6], label="颈椎"), Patch(color=VERT_COLORS[13], label="胸椎"),
               Patch(color=VERT_COLORS[22], label="腰椎")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, labelcolor=TEXT, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(ASSETS / "viz_render3d.png", facecolor=PLATE)
    plt.close(fig)
    return verts, faces, cols


# ---------------------------------------------------------------- 6. overlay styles
def fig_overlay():
    xi = int(round(np.median(PTS[:, 0])))
    img = sag(CTW, xi)
    lab = sag(SEG, xi)
    z0 = int(PTS[LABELS == 20][0][2]) - 25
    z1 = int(PTS[LABELS == 24][0][2]) + 25
    z0, z1 = min(z0, z1), max(z0, z1)
    rows = slice(max(z0 - 60, 0), z1 + 20)
    img, lab = img[rows], lab[rows]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.9), facecolor=PLATE, gridspec_kw=dict(wspace=0.03))
    titles = ["不透明填充：挡住骨小梁", "半透明填充 α=0.4", "仅轮廓：纹理可见", "轮廓 + 编号：最适合打印"]
    for n, (ax, t) in enumerate(zip(axes, titles)):
        ax.imshow(img, cmap="gray", origin="lower", aspect=SP[2] / SP[1])
        if n in (0, 1):
            ax.imshow(np.ma.masked_equal(lab, 0), cmap=label_cmap(), vmin=0, vmax=28, origin="lower",
                      aspect=SP[2] / SP[1], alpha=1.0 if n == 0 else 0.4, interpolation="nearest")
        if n in (2, 3):
            overlay_contours(ax, lab, lw=1.1)
        if n == 3:
            for lbl in np.unique(lab):
                if lbl == 0:
                    continue
                rr, cc = np.nonzero(lab == lbl)
                ax.text(cc.min() - 6, rr.mean(), VERT_NAMES[int(lbl)], color=VERT_COLORS[int(lbl)],
                        fontsize=8, ha="right", va="center", fontweight="bold")
        plate_axes(ax)
        ax.set_xlim(20, img.shape[1] - 10)
        ax.set_title(t, color=TEXT, fontsize=8.5, loc="left")
    fig.savefig(ASSETS / "viz_overlay.png", facecolor=PLATE)
    plt.close(fig)


# ---------------------------------------------------------------- 7. one-slide template
def fig_template(mesh):
    verts, faces, cols = mesh
    W, H = 13.333, 7.5
    fig = plt.figure(figsize=(W, H), facecolor="#F3F5F4")
    fig.text(0.035, 0.94, "病例 sub-verse404 · VerSe'19 · 胸腰段 CT（C7–L5，18 节椎骨）", fontsize=17, color=INK,
             fontweight="bold")
    fig.text(0.035, 0.905, "左：三维概览定位 · 中：CPR 拉直看整体序列 · 右：逐椎体横断面看细节。所有切片共用同一套椎体颜色。",
             fontsize=10.5, color=SLATE)

    crop = render_view(verts, faces, cols, 14, 135, np.array([-1, 1, 0.6]), "#F3F5F4")
    ax3 = fig.add_axes([0.03, 0.08, 0.22, 0.78])
    ax3.imshow(crop)
    ax3.set_axis_off()
    fig.text(0.035, 0.045, "三维表面重建（左前斜）", fontsize=9.5, color=INK)

    s_img, s_lab, st = cpr("sag")
    c_img, c_lab, _ = cpr("cor")
    for x0, im, lb, t, marks in [(0.27, s_img, s_lab, "CPR 矢状", ("S", "I", "A", "P")),
                                 (0.415, c_img, c_lab, "CPR 冠状", ("S", "I", "R", "L"))]:
        ax = fig.add_axes([x0, 0.06, 0.14, 0.82])
        ax.imshow(im, cmap="gray", origin="upper", aspect="auto")
        overlay_contours(ax, lb, lw=0.7)
        plate_axes(ax)
        orient_marks(ax, *marks)
        ax.set_title(t, fontsize=9.5, color=INK, loc="left")
        if x0 < 0.3:
            scale_bar(ax, 1 / st, 50)
    axc = fig.axes[-1]
    for k, lbl in enumerate(LABELS):
        r = (ARC_CTD[k] - S[0]) / (S[1] - S[0])
        axc.text(c_img.shape[1] + 4, r, VERT_NAMES[int(lbl)], color=INK, fontsize=7.5, va="center", clip_on=False)

    keys = [k for k, l in enumerate(LABELS) if VERT_NAMES[int(l)] in ("T4", "T8", "T12", "L1", "L3", "L5")]
    h = 0.34
    w = h * H / W * (97 / 101)
    gx = 0.60
    for n, k in enumerate(keys):
        r, c = divmod(n, 3)
        ax = fig.add_axes([gx + c * (w + 0.01), 0.49 - r * (h + 0.06), w, h])
        img, lab = oblique(k)
        ax.imshow(img, cmap="gray")
        lbl = LABELS[k]
        for cc in measure.find_contours((lab == lbl).astype(float), 0.5):
            ax.plot(cc[:, 1], cc[:, 0], color=VERT_COLORS[int(lbl)], lw=1.1)
        plate_axes(ax)
        ax.set_title(VERT_NAMES[int(lbl)], fontsize=10, color=INK, loc="left", fontweight="bold")
    fig.text(0.60, 0.885, "关键节段横断面（沿中心线法向重切，A↑ R←）", fontsize=9.5, color=INK)
    fig.text(0.60, 0.035, "颜色：颈椎蓝 · 胸椎青绿 · 腰椎赭黄；比例尺 50 mm；窗宽/窗位 1800/400", fontsize=8.5, color=SLATE)
    fig.savefig(ASSETS / "viz_template.png", facecolor=fig.get_facecolor(), bbox_inches=None, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 8. sprite for the web slice scrubber
def sprite():
    X = CT.shape[0]
    frames, frames_l = [], []
    cm = label_cmap()
    for i in range(X):
        g = sag(CTW, i)
        lab = sag(SEG, i)
        g = zoom(g, (SP[2] / SP[1], 1), order=1)
        lab = zoom(lab, (SP[2] / SP[1], 1), order=0)
        g8 = (np.flipud(g) * 255).astype(np.uint8)
        rgb = np.repeat(g8[..., None], 3, axis=2).astype(np.float32)
        col = (cm(np.flipud(lab))[..., :3] * 255).astype(np.float32)
        a = (np.flipud(lab) > 0)[..., None] * 0.5
        frames.append(Image.fromarray(g8))
        frames_l.append(Image.fromarray((rgb * (1 - a) + col * a).astype(np.uint8)))
    h = 520
    w = int(frames[0].width * h / frames[0].height)
    cols = 12
    rows = int(np.ceil(X / cols))
    meta = dict(n=X, cols=cols, w=w, h=h, mm_per_slice=float(SP[0]))
    for name, fr in [("sprite_ct.jpg", frames), ("sprite_lab.jpg", frames_l)]:
        sheet = Image.new("RGB", (cols * w, rows * h))
        for i, f in enumerate(fr):
            r, c = divmod(i, cols)
            sheet.paste(f.convert("RGB").resize((w, h), Image.BILINEAR), (c * w, r * h))
        sheet.save(ASSETS / name, quality=80)
    (ASSETS / "sprite.json").write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    fig_windows(); print("windows")
    fig_projections(); print("proj")
    fig_lightbox(); print("lightbox")
    fig_cpr(); print("cpr")
    fig_overlay(); print("overlay")
    mesh = fig_render3d(); print("render", mesh[1].shape)
    fig_template(mesh); print("template")
    print(sprite())
