# Food-101 image classification

End-to-end **computer vision** experiment on the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset: exploratory analysis, stratified train/validation split, `tf.data` pipelines, and **transfer learning** with **EfficientNet-B0** (ImageNet weights) plus a small classification head.

The main artifact is the Jupyter notebook [`food-classification-model.ipynb`](food-classification-model.ipynb). The notebook was developed on **Kaggle** (GPU); paths and outputs reflect that environment, with **local runs** supported via `FOOD101_IMAGES_DIR` or a standard folder layout (see below).

## What this demonstrates

- Framing a multi-class problem (101 balanced classes, 1,000 images each)
- Building file lists and label indices in pandas
- Stratified splitting with scikit-learn
- Efficient batched loading with TensorFlow
- Transfer learning: frozen backbone, trainable head, optional callbacks for fine-tuning experiments

## Quick start

### 1. Environment

Use **Python 3.11 or 3.12**. TensorFlow does not publish wheels for very new interpreters (for example **3.14**); if `python3` is too new on your machine, create the venv with 3.11 explicitly:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

Download Food-101 and point the notebook at the directory that contains **one subfolder per class** (the `images` folder in the official layout).

**Option A — environment variable (any layout):**

```bash
export FOOD101_IMAGES_DIR="/absolute/path/to/food-101/images"
```

**Option B — convention (no env var):** put the `images` folder at either:

- `data/food-101/images`, or  
- `food-101/images`  

(relative to the repo root or your current working directory when you launch Jupyter).

**Kaggle:** attach the Food-101 dataset; the notebook’s `find_food101_images_dir()` still searches under `/kaggle/input` when local paths are not found.

### 3. Run

```bash
jupyter lab food-classification-model.ipynb
```

Run cells from the top. The notebook includes exploratory sections and multiple training experiments; later cells repeat setup for iterative runs on Kaggle.

## Project layout

| Path | Purpose |
|------|--------|
| `food-classification-model.ipynb` | EDA, data pipeline, EfficientNet-B0 model, training |
| `requirements.txt` | Python dependencies |
| `docs/hackathon-notes.md` | Original study / hackathon notes (preserved) |

## CI

GitHub Actions installs dependencies and verifies imports (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Study notes

Hackathon-style notes and interview prompts that lived in the root README are kept in [`docs/hackathon-notes.md`](docs/hackathon-notes.md).

## License

See [LICENSE](LICENSE).
