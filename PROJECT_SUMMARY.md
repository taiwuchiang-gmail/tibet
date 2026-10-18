# Tibetan Woodblock OCR — Project Summary

Handoff summary for starting a Claude Code Project. Written 2026-10-17.
Project folder: `C:\Users\tchiang\code\tibet` (MATLAB, runs locally; needs the local MATLAB install).

## Goal
Recognize glyphs/syllables in Degé Kangyur woodblock scans (Tibetan script):
segment pages into glyph crops, learn identity-discriminative features without labels,
then find a way to label them.

## What's built
- **`segment_page.m`** — page-frame detection, adaptive glyph-size filtering, rule-line removal
  (morphological opening), line-band detection via row-projection valleys.
- **Feature learning** — conv autoencoder (conv+BN+ReLU+maxpool ×3, global-average-pool bottleneck,
  transposed-conv decoder), then NT-Xent contrastive fine-tuning with a projection head
  (`ntxent_loss.m`, `contrastive_model_loss.m`, `augment_glyph.m`, `make_contrastive_batch_img.m`).
  The result, **`netCC`**, is the best encoder so far.
- **`run_tibet_pipeline.m`** — reproduces the verified pipeline end to end
  (segmentation → conv AE → contrastive → invariance test → t-SNE + cluster montages),
  writing `results/<timestamp>/` (metrics.txt, figures, models.mat). `cfgOverride = struct('quick', true)` for a smoke test.
- Helpers: `load_crop_dataset.m`, `render_tsne_canvas.m`, `plot_feature_tsne.m`, `pad_resize_square.m`, `group_by_proximity.m` (abandoned approach).

## Latest full run — `results/20261017_122904` (CPU)
- 853 clean crops (printed transcription); conv AE recon MSE 0.0133; contrastive final loss 0.729
- Invariance (cosine similarity to unperturbed glyph):

| model | shift mean | shift worst | scale mean | scale worst |
|---|---|---|---|---|
| Conv AE (recon only) | 0.951 | 0.869 | 0.934 | 0.830 |
| Conv + contrastive (netCC) | 0.992 | 0.973 | 0.987 | 0.959 |

## GPU / single precision (2026-10-18, RTX A3000 6 GB, MATLAB R2026a)
- All computation is `single`; GPU arrays are used wherever the functions support them.
  Results match the CPU baseline: run `results/20261018_013102` gave recon MSE 0.0131, contrastive loss 0.741,
  netCC shift 0.992 / 0.979, scale 0.988 / 0.965.
- `augment_glyph.m` is now batched (one bilinear warp for the whole minibatch on the GPU): contrastive
  batch build 290 ms → 8 ms, 60 contrastive epochs in ~28 s.
- `segment_page.m` (`'UseGPU'` option) reproduces all 12,994 existing crops bit-for-bit. Only `imopen` and `bwdist` run on the GPU;
  `imbinarize`, Canny/`hough`, `bwconncomp`, `bwareaopen` have no gpuArray support, and gpuArray `imrotate` was rejected because
  it rounds differently and changes components. Segmentation is CPU/PNG-write bound either way.
- No gpuArray support either: `tsne`, `imtranslate` (single), grayscale `imdilate`/`imerode`. `bwareaopen` needs a double P.

## Key findings
1. Segmentation fragmentation on real scans is a **scan-quality** problem (faint/broken strokes), not a bug —
   clean printed text segments into one box per syllable.
2. Deep stacked sigmoid AEs collapse (vanishing gradients). ReLU+BN fixes collapse but not identity separation on noisy scans.
3. **Input quality was the dominant lever**: on clean crops even a shallow AE clusters same glyphs; conv + contrastive is best.
4. Tibetan rendering in MATLAB works with **Noto Serif Tibetan / Jomolhari installed for all users** (system-wide),
   then `text(..., 'FontName', 'Noto Serif Tibetan')`. Microsoft Himalaya is broken; Canvas 2D `fillText` is broken.
5. Neither Claude nor the user reads Tibetan — **never validate by eye**. Use pixel-diff vs a trusted reference,
   codepoint dumps, or the **shad trick** (shad ། is a plain vertical stroke; check crops labeled shad really are).

## Open problem: labeling
- **Alignment-based labeling (ground-truth text zipped onto crops in reading order) — BLOCKED.**
  Bounding-box-proximity grouping gave 345 groups vs 346 tokens, but the shad check showed only ~2/24 correct. Abandoned.
- **Planned replacement:** column-projection syllable segmentation on raw ink within each line,
  validated first on a single non-wrapping line (`single_line_test.png`, from `scratch/ground_truth_snippet_v2.txt`).
- **Alternative (user's idea):** K-means on unlabeled netCC embeddings → render labeled reference glyphs one at a time →
  embed with netCC → nearest centroid → propagate label to the cluster. Risk: a rendered-vs-real domain gap
  already failed once (shuffled control ≈ 0.87 similarity either way), so verify empirically.

## Working rules
- Verify each step before building the next — no one-shot pipelines.
- MATLAB figures: use `figure(1)`, `figure(2)`, … so plots don't overwrite each other.
- Don't retype Tibetan Unicode by hand; extract it mechanically and verify codepoints.
