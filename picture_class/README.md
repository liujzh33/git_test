# 公开脊柱影像数据集调研（网页版）

一份约 15 分钟的网页讲稿，覆盖五个公开脊柱影像数据集：VerSe、CTSpine1K、SPIDER、RSNA 2024 腰椎、CSXA。页面里的图已经生成好，放在 `site/assets`，打开就能讲，不必再下载数据。

## 目录

```
picture_class/
  site/          网页：index.html、style.css、app.js、assets/
  scripts/       下载样例、检查数据、重新出图
  requirements.txt
```

原始影像没有放进仓库。五个数据集只抽了样例，合计大约几百 MB，用 `scripts/download_samples.py` 按需下载到 `data/`（该目录已被 gitignore）。

## 直接放映

需要本机有 Python 3。在 `picture_class` 目录下启动一个静态服务器，再用浏览器打开。不要双击 `index.html`：逐层翻切片那一页要用相对路径读图，用 `file://` 打开会失败。

Windows 上如果 `python` 不在 PATH 里，换成 Python 的完整路径。这台机器是：

```
c:\Users\Administrator\Desktop\.python313\python.exe -m http.server 8765
```

一般环境：

```
cd picture_class
python -m http.server 8765
```

浏览器打开 <http://127.0.0.1:8765/index.html>。

放映按键：

| 键 | 作用 |
| --- | --- |
| ← →、空格、PageUp / PageDown | 翻页 |
| Home / End | 第一页 / 最后一页 |
| N | 显示或隐藏本页讲稿 |
| F | 全屏 |

左侧是按椎骨形状排的导航，颜色分颈、胸、腰，点击可跳页。

## 重新出图（可选）

图已经在 `site/assets`。只有要改图或换样例时才需要下面这些步骤。

安装依赖（在 `picture_class` 目录）：

```
python -m pip install -r requirements.txt
```

中文图用微软雅黑。没有这套字体时，matplotlib 会退到 DejaVu Sans，中文会变成方框。

下载样例。脚本用 HTTP Range 从压缩包里只取选中的病例，不会下完整数据集。Hugging Face 走 `hf-mirror.com`。

```
cd scripts
python download_samples.py
```

也可以只下一个：`python download_samples.py verse`。可选名字：`verse`、`spider`、`ctspine1k`、`rsna`、`csxa`。

各数据集实际取的内容：

| 数据集 | 样例 |
| --- | --- |
| VerSe'19 | sub-verse010、verse209、verse256（后两例是同一病人拆开的两段） |
| CTSpine1K | MSD 肝脏 `liver_0`（层厚 5 mm）、COVID-19 `volume-covid19-A-0003`（层厚 1 mm），含分割 |
| SPIDER | `overview.csv`、`radiological_gradings.csv`，以及 1_t1、1_t2、100_t2 的图像和掩膜 |
| RSNA 2024 | Hugging Face 镜像的 `train.parquet`（约 345 MB，30,022 张 128×128 图），不是 Kaggle 原始 DICOM |
| CSXA | `datasets.xlsx`，以及均匀抽的 8 张 PNG 和对应关键点 JSON |

CTSpine1K 的 COVID 标签在镜像上的文件名是 `volume-covid19-A-0003_ct_seg.nii.gz`，脚本会把它存成本地的 `volume-covid19-A-0003_seg.nii.gz`。

下载完成后，在 `scripts` 目录重新出图，结果写回 `site/assets`：

```
python fig_datasets.py
python fig_metrics.py
python fig_viz.py
```

`fig_datasets.py` 里的 RSNA 例图会读 `data/rsna2024/ac.pkl`（每张图的邻像素相关系数，用来剔掉镜像里的噪声图）。这个文件没有放进仓库。没有它时，`fig_datasets.py` 在 RSNA 那一步会报错，其余数据集的图不受影响。页面上的 `rsna_grid.png`、`rsna_quality.png` 已经是筛过的结果，放映不依赖这个文件。

`inspect_basic.py` 把样例的形状、间距、值域写到 `basic_report.json`，只在下载完样例之后才有意义。

`shots.ps1` 用本机 Edge 无头模式把每一页截成 1600×900 的图，输出到 `shots/`。需要先开着上面的服务器。这是检查排版用的，放映不需要。

## 页面上的数字从哪来

灰度、形状、间距、样例图来自下载的样例。排行榜和论文指标没有在本地重新训练，出处写在网页最后一页。RSNA 用的是 Hugging Face 镜像：大约 2% 的图基本是噪声，网页里单独有一页说明。
