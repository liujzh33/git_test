"""Print shape / spacing / orientation / intensity range / label set for each sample."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

D = Path(__file__).resolve().parents[1] / "data"
report = {}


def stats(a):
    a = np.asarray(a)
    p = np.percentile(a, [0.5, 50, 99.5])
    return dict(dtype=str(a.dtype), min=float(a.min()), max=float(a.max()),
                p0_5=float(p[0]), median=float(p[1]), p99_5=float(p[2]))


def nii(path, seg=None):
    img = nib.load(path)
    a = np.asanyarray(img.dataobj)
    r = dict(file=path.name, shape=list(a.shape),
             spacing=[round(float(z), 3) for z in img.header.get_zooms()[:3]],
             orientation="".join(nib.aff2axcodes(img.affine)), **stats(a))
    if seg is not None and seg.exists():
        s = np.asanyarray(nib.load(seg).dataobj).astype(np.int32)
        labs, cnt = np.unique(s, return_counts=True)
        r["labels"] = {int(l): int(c) for l, c in zip(labs, cnt)}
        r["seg_shape"] = list(s.shape)
    return r


def mha(path, seg=None):
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(img)  # z, y, x
    d = img.GetDirection()
    orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(d)
    r = dict(file=path.name, shape_zyx=list(a.shape), size_xyz=list(img.GetSize()),
             spacing_xyz=[round(s, 3) for s in img.GetSpacing()], orientation=orient, **stats(a))
    if seg is not None and seg.exists():
        s = sitk.GetArrayFromImage(sitk.ReadImage(str(seg)))
        labs, cnt = np.unique(s, return_counts=True)
        r["labels"] = {int(l): int(c) for l, c in zip(labs, cnt)}
    return r


v = D / "verse19"
report["verse"] = []
for ct in sorted((v / "rawdata").rglob("*_ct.nii.gz")):
    stem = ct.name.replace("_ct.nii.gz", "")
    der = v / "derivatives" / ct.parent.name
    r = nii(ct, der / f"{stem}_seg-vert_msk.nii.gz")
    ctd = json.loads((der / f"{stem}_seg-subreg_ctd.json").read_text())
    r["centroids_head"] = ctd[:3]
    r["n_centroids"] = len(ctd) - 1
    report["verse"].append(r)

c = D / "ctspine1k"
report["ctspine1k"] = []
for vol in sorted(c.glob("*/*.nii.gz")):
    if "_seg" in vol.name:
        continue
    stem = vol.name[:-len(".nii.gz")].removesuffix("_ct")
    seg = vol.with_name(f"{stem}_seg.nii.gz")
    report["ctspine1k"].append(nii(vol, seg))

s = D / "spider"
report["spider"] = []
for im in sorted((s / "images").glob("*.mha")):
    report["spider"].append(mha(im, s / "masks" / im.name))

out = Path(__file__).resolve().parents[1] / "basic_report.json"
out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps(report, indent=1, ensure_ascii=False))
