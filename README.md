# Tibetan Woodblock Glyph Recognition (MATLAB)

Unsupervised feature learning for glyphs in Degé Kangyur woodblock scans.
The pipeline segments pages into glyph crops and learns features that put
copies of the same glyph close together without any labels: a convolutional
autoencoder followed by contrastive fine-tuning. The next goal is a way to
label the resulting clusters.

Two digit datasets (printed fonts and handwritten MNIST) act as labeled
sanity checks: the features are learned the same way, and the known labels
are only used to score how well the features separate the classes.

## Requirements

- MATLAB R2026a (developed on R2026a; older releases are untested)
- Deep Learning Toolbox (`dlnetwork`, `trainnet`, custom training loop)
- Image Processing Toolbox (segmentation, morphology, `imresize`)
- Statistics and Machine Learning Toolbox (`tsne`, `kmeans`)
- Computer Vision Toolbox (`insertText`, only for building the printed-digit set)
- Parallel Computing Toolbox and a CUDA GPU (optional; used automatically when
  `canUseGPU()` is true, otherwise everything runs on the CPU)

All computation is single precision. On an RTX A3000 Laptop GPU (6 GB), a full
Tibetan run takes about 1.5 minutes and a full MNIST run about 3 minutes.

## Quick start

From the project folder in MATLAB:

```matlab
run_tibet_pipeline                                        % Tibetan, full run

cfgOverride = struct('quick', true);                      % smoke test, a few epochs
run_tibet_pipeline

cfgOverride = struct('dataset', 'digits_printed');        % 0-9 in every installed font
run_tibet_pipeline

cfgOverride = struct('dataset', 'digits_mnist');          % handwritten MNIST test set
run_tibet_pipeline
```

`cfgOverride` is read once per run, so set it before each call. Every run
writes `results/<timestamp>[_<dataset>...]/` containing `metrics.txt`, the
figures as PNG, `models.mat` (trained networks, features, t-SNE coordinates)
and `run_log.txt`. Figures stay open; each dataset uses its own figure numbers
(Tibetan 1-8, printed digits 101-108, MNIST 201-208).

The digit datasets are built under `digits/` on first use:
`make_printed_digits.m` renders 0-9 in every usable installed TrueType font
(about 255 fonts on Windows), and `make_mnist_digits.m` downloads the MNIST
test set.

## Pipeline

1. **Data.** Tibetan: `segment_page.m` finds the page frame, removes rule
   lines, deskews, splits lines by row-projection valleys and crops
   connected-component glyphs. Inputs are `1-1-2b.jpg`,
   `samples_adarshah/*.jpg` (woodblock scans) and
   `printed_transcription_test.png` (a screenshot of the clean digital
   transcription from the Adarshah reader). The clean crops are what the
   networks train on.
2. **Optional FC baseline** (`cfg.runFCBaseline`, Tibetan only): a fully
   connected autoencoder, kept to reproduce the conv-vs-FC invariance comparison.
3. **Convolutional autoencoder** on 32x32 crops: three conv+BN+ReLU(+maxpool)
   stages, a pooled bottleneck, and a transposed-conv decoder.
4. **Contrastive fine-tuning** of the encoder with a projection head and the
   NT-Xent loss on two random augmentations of each glyph (rotation,
   translation, scale, stroke thickness; `augment_glyph.m`). The result is
   `netCC`.
5. **Invariance test:** cosine similarity between a glyph's features and those
   of the same glyph shifted by 1-3 px or scaled by 0.85-1.15x.
6. **t-SNE** of the `netCC` features with glyph thumbnails, plus montages of
   the densest clusters. Digits also get a t-SNE with ink colored by true
   label, and label scores: leave-one-out 5-NN accuracy, K-means (k=10)
   purity and NMI, and what else ends up in the cluster holding most 0s.

### Main options (`cfgOverride` fields)

| field | default | meaning |
|---|---|---|
| `dataset` | `'tibetan'` | `'tibetan'`, `'digits_printed'` or `'digits_mnist'` |
| `quick` | `false` | a few epochs and at most 2000 images, for smoke tests |
| `resegment` | `true` | Tibetan: re-run page segmentation (`false` reuses the crop folders) |
| `bottleneck` | `'grid2'` | `'grid2'`: 2x2 average pool of the final 8x8x32 map (128-d); `'gap'`: global average pool (32-d) |
| `labelsPerClass` | `0` | digits: train with the semi-supervised SupCon loss using this many known labels per class |
| `convEpochs`, `contrastEpochs`, `contrastBatch`, `tau` | see script | training settings |
| `seed` | `1` | random seed (CPU and GPU) |

## Results

Tibetan, clean printed crops (853 glyphs), cosine similarity of `netCC`
features under perturbation (mean / worst case):

| bottleneck | shift | scale |
|---|---|---|
| `'gap'` | 0.992 / 0.977 | 0.985 / 0.957 |
| `'grid2'` | 0.975 / 0.941 | 0.968 / 0.907 |

Digits, `netCC` features. The "0-cluster" is the K-means cluster that holds
the most 0s, listed as 0s / 6s / 9s / other digits in it:

| dataset | bottleneck | 5-NN accuracy | K-means purity | 0-cluster |
|---|---|---|---|---|
| MNIST | `'gap'` | 0.965 | 0.445 | 380 / 134 / 85 / 272 |
| MNIST | `'grid2'` | 0.973 | 0.672 | 861 / 24 / 5 / 54 |
| printed | `'gap'` | 0.960 | 0.667 | 143 / 39 / 37 / 19 |
| printed | `'grid2'` | 0.974 | 0.855 | 220 / 6 / 4 / 4 |

Global average pooling records which stroke patterns a glyph contains but
not where they are, so 0, 6 and 9 (all built from loops and curves) looked
alike. Pooling per quadrant keeps that layout, at a small cost in shift
invariance. Adding 1-50 known labels per digit through the SupCon loss
improved MNIST 5-NN accuracy only slightly (0.9745 to 0.978, mean of 3 seeds)
and did not reliably improve the clusters. Details are in `PROJECT_SUMMARY.md`.

## Files

| file | purpose |
|---|---|
| `run_tibet_pipeline.m` | end-to-end pipeline and configuration |
| `segment_page.m` | page segmentation into glyph crops |
| `load_crop_dataset.m`, `pad_resize_square.m` | load crops as padded 32x32 single images |
| `augment_glyph.m` | batched random augmentation on CPU or GPU |
| `make_contrastive_batch_img.m`, `make_contrastive_batch.m` | two augmented views per sample |
| `ntxent_loss.m`, `supcon_loss.m`, `contrastive_model_loss.m` | contrastive losses and gradient step |
| `label_metrics.m` | kNN accuracy, K-means purity / NMI |
| `render_tsne_canvas.m`, `plot_feature_tsne.m` | t-SNE plots with glyph thumbnails |
| `make_printed_digits.m`, `make_mnist_digits.m` | build the digit datasets |
| `cc.m`, `line_intersect_coord.m`, `group_by_proximity.m` | older helpers (connected components, proximity grouping) |
| `PROJECT_SUMMARY.md` | findings, open problems and working notes |
| `scratch/` | one-off verification and diagnosis scripts |

## Status and open problems

- Labeling glyphs by aligning the ground-truth text with crops in reading
  order did not work (about 2 of 24 spot checks correct) and was dropped.
- Planned next: column-projection syllable segmentation within each line, and
  labeling by K-means on `netCC` features plus rendered reference glyphs.
- Real scans fragment into broken strokes because of scan quality, so the
  networks currently train on the clean transcription crops.

The woodblock scans and the transcription screenshot come from the
[Adarshah](https://adarshah.org) digital Kangyur.
