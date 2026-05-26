# Changelog

## v1.0.0 — 2026-05-25

### Added
- 10 adversarial attack engines using real ResNet50 gradients
  - FGSM, PGD, LSB, DCT, C&W, DeepFool, AutoPGD, Patch, Gaussian, JSMA
- PyQt6 GUI with 4 tabs: Attack, Benchmark, Heatmap, Settings
- Live benchmark runner comparing all 10 engines with matplotlib charts
- Pixel difference heatmap with 3 visualisation modes
- Settings panel with auto-save and keyboard shortcuts
- Full CLI: imageecho run, imageecho benchmark, imageecho batch
- Batch processing — entire folders with any engine
- unOptimal() binary search for strongest invisible attack
- 48 pytest tests covering engines, context, and surrogate model
- GitHub Actions CI — tests + lint on every push
- Professional documentation: ENGINES.md, ARCHITECTURE.md, CONTRIBUTING.md

### Technical
- Real neural network gradients via ResNet50 backpropagation
- Transfer attacks that fool VGG, EfficientNet, and cloud APIs
- SSIM + PSNR metrics on every attack
- Dark theme PyQt6 interface with drag-and-drop
