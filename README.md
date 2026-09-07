# Panorama Generation Project

Group G16 — Project P09 | CO5430 Computer Vision

<p align="center">
  <img src="docs/images/1.jpg" alt="Panoramic view" width="50%" />
</p>

## Team Members
| ID | Name |
|---|---|
| E/22/054 | [Danuja Nirodhana](https://people.ce.pdn.ac.lk/students/e22/054/) |
| E/22/184 | [Bhagya Karunanayake](https://people.ce.pdn.ac.lk/students/e22/184/) |
| E/22/179 | [Janith Kahagalla](https://people.ce.pdn.ac.lk/students/e22/179/) |
| E/22/205 | [Ashen Kumarasinghe](https://people.ce.pdn.ac.lk/students/e22/205/) |

---

## Project Structure

```
panorama-generation/
├── data/
│   ├── raw/               ← original handheld captures (image1.jpeg, image2.jpeg, frame_*.jpg)
│   └── processed/         ← resized/normalized copies (optional)
├── src/
│   ├── features/          ← SIFT / ORB extraction
│   ├── matching/          ← FLANN + Lowe's ratio test
│   ├── geometry/          ← RANSAC homography + M4 affine fallback
│   ├── stitching/         ← canvas allocation, warping, compositing
│   ├── blending/          ← M4: feathering & Laplacian pyramid blending
│   ├── bundle_adjustment/ ← M4: global pose optimization
│   ├── evaluation/        ← RMSE, inlier ratio, seam metrics
│   └── pipeline.py        ← top-level CLI orchestrator
├── configs/
│   └── default.yaml       ← all tuneable thresholds
├── notebooks/             ← exploratory notebooks
├── outputs/
│   ├── m2/                ← M2 ORB baseline results
│   ├── m3/                ← M3 SIFT+FLANN results
│   ├── m4/                ← M4 results (feather / multiband)
│   └── metrics/           ← benchmark CSVs, fallback log, BA report
├── tests/
│   ├── test_features.py
│   ├── test_matching.py
│   ├── test_geometry.py
│   ├── test_blending.py
│   └── benchmark.py
├── docs/
│   ├── M2_baseline.md
│   ├── M3_sift_flann.md
│   └── M4_planned_improvements.md
├── generate_dummy_data.py ← synthetic 4-frame test sequence
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd panorama-generation

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) generate synthetic 4-frame test images
python generate_dummy_data.py
```

---

## Running the Pipeline

### Two-image stitching (M3 default — hard overwrite)
```bash
python src/pipeline.py
```

### Two-image stitching with M4 feathered blending
```bash
python src/pipeline.py --blend feather
```

### Two-image stitching with M4 pyramid blending
```bash
python src/pipeline.py --blend multiband
```

### Multi-image stitching (all images in data/raw/, sorted order)
```bash
python src/pipeline.py --multi --blend feather
```

### Multi-image + global bundle adjustment
```bash
python src/pipeline.py --multi --blend multiband --ba
```

### Custom dataset / output directory
```bash
python src/pipeline.py --data path/to/images/ --out path/to/output/ --blend feather
```

### Run benchmark suite
```bash
python tests/benchmark.py               # default (overwrite mode)
python tests/benchmark.py --blend feather
```

### Run unit tests
```bash
pytest tests/ -v
```

---

## Results — M2 → M3 → M4

| Metric | M2 (ORB+BF) | M3 (SIFT+FLANN) | M4 (feather) | M4 (multiband+BA) |
|---|:---:|:---:|:---:|:---:|
| Inlier ratio | N/A | **76.56 %** | 76.56 % | 76.56 % |
| RMSE (px) | N/A | **0.54** | 0.54 | ≤0.54 (BA reduces drift) |
| Speed (s/pair) | ~0.2 | **~0.65** | ~0.70 | ~0.95 |
| Seam quality | Hard seam | Hard seam | **Smooth** | **Sharp + seamless** |
| N-image | ❌ | ✅ | ✅ | ✅ |
| Bundle adjustment | ❌ | ❌ | optional | **✅** |
| Catastrophic warp protection | ❌ | ❌ | **✅ (1.3× fallback)** | **✅** |

> Full per-pair metrics available in `outputs/metrics/benchmark_*.csv` after running benchmark.

---

## Configuration

Edit `configs/default.yaml` to change any threshold without touching source code:

```yaml
ratio_thresh:        0.70    # Lowe's ratio test
stretch_limit:       1.3     # Affine fallback trigger
blend_mode:          feather # overwrite | feather | multiband
compensate_exposure: false
```

---

## Dataset
Dataset source: https://sourceforge.net/adobe/adobedatasets/home/Home/
