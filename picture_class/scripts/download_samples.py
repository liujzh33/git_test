"""Download a few representative cases from five public spine datasets.

Large archives (VerSe, SPIDER, CSXA) are read member-by-member with HTTP
Range requests, so only the selected cases are transferred.
"""
import sys
from pathlib import Path

import requests

from httpzip import open_remote_zip

ROOT = Path(__file__).resolve().parents[1] / "data"
HF = "https://hf-mirror.com/datasets"


def fetch(url, dst, chunk=1 << 20):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        print("skip", dst.name)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk):
                f.write(c)
    tmp.replace(dst)
    print("ok  ", dst.name, f"{dst.stat().st_size/1e6:.1f} MB")
    return dst


def extract(zf, names, outdir, strip=0):
    outdir = Path(outdir)
    for name in names:
        rel = Path(*Path(name).parts[strip:])
        dst = outdir / rel
        if dst.exists():
            print("skip", rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(zf.read(name))
        print("ok  ", rel, f"{dst.stat().st_size/1e6:.1f} MB")


def verse():
    url = "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip"
    zf = open_remote_zip(url)
    keys = ["sub-verse010_", "sub-verse404_split-verse209_", "sub-verse404_split-verse256_"]
    names = [i.filename for i in zf.infolist()
             if not i.filename.startswith("__MACOSX") and not i.filename.endswith("/")
             and any(k in i.filename for k in keys)]
    extract(zf, names, ROOT / "verse19", strip=1)


def spider():
    rec = "https://zenodo.org/api/records/10159290/files"
    out = ROOT / "spider"
    for f in ["overview.csv", "radiological_gradings.csv"]:
        fetch(f"{rec}/{f}/content", out / f)
    want = {"images/1_t1.mha", "images/1_t2.mha", "images/100_t2.mha"}
    zf = open_remote_zip(f"{rec}/images.zip/content")
    extract(zf, [n for n in zf.namelist() if n in want], out)
    mz = open_remote_zip(f"{rec}/masks.zip/content")
    mnames = {n.replace("images/", "masks/") for n in want}
    extract(mz, [n for n in mz.namelist() if n in mnames], out)


def ctspine1k():
    base = f"{HF}/alexanderdann/CTSpine1K/resolve/main"
    out = ROOT / "ctspine1k"
    for f in ["metadata/data_split.txt", "metadata/path.csv"]:
        fetch(f"{base}/{f}", out / f)
    cases = [("MSD-T10", "liver_0"), ("COVID-19", "volume-covid19-A-0003_ct")]
    for src, cid in cases:
        fetch(f"{base}/raw_data/volumes/{src}/{cid}.nii.gz", out / src / f"{cid}.nii.gz")
    fetch(f"{base}/raw_data/labels/MSD-T10/liver_0_seg.nii.gz", out / "MSD-T10" / "liver_0_seg.nii.gz")
    # 镜像上的文件名是 _ct_seg，本地仍存成 _seg，和出图脚本一致。
    fetch(f"{base}/raw_data/labels/COVID-19/volume-covid19-A-0003_ct_seg.nii.gz",
          out / "COVID-19" / "volume-covid19-A-0003_seg.nii.gz")


def rsna():
    base = f"{HF}/GilbertKrantz/rsna-lumbar-spine-dataset/resolve/main"
    fetch(f"{base}/data/train-00000-of-00001.parquet", ROOT / "rsna2024" / "train.parquet")


def csxa():
    base = f"{HF}/MartialTerran/Cervical_Spine_X-ray_Atlas_CSXA_V3.0/resolve/main"
    out = ROOT / "csxa"
    fetch(f"{base}/datasets.xlsx", out / "datasets.xlsx")
    zp = open_remote_zip(f"{base}/datasets-PNG.zip")
    pngs = [n for n in zp.namelist() if n.lower().endswith(".png")]
    (out / "png_list.txt").write_text("\n".join(pngs), encoding="utf-8")
    pick = pngs[:: max(1, len(pngs) // 8)][:8]
    extract(zp, pick, out)
    zj = open_remote_zip(f"{base}/datasets-JSON.zip")
    jsons = [n for n in zj.namelist() if n.lower().endswith(".json")]
    (out / "json_list.txt").write_text("\n".join(jsons), encoding="utf-8")
    stems = {Path(p).stem for p in pick}
    extract(zj, [n for n in jsons if Path(n).stem in stems], out)


if __name__ == "__main__":
    jobs = {"verse": verse, "spider": spider, "ctspine1k": ctspine1k, "rsna": rsna, "csxa": csxa}
    for name in sys.argv[1:] or jobs:
        print(f"== {name}")
        try:
            jobs[name]()
        except Exception as e:
            print("FAILED", name, repr(e))
