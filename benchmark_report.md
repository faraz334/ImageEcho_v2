# ImageEcho Benchmark Report

| Engine | SSIM | PSNR | Mean ? | Fooled |
|--------|------|------|--------|--------|
| fgsm | 0.9943 | 30.1 | 7.90 | YES |
| pgd | 0.9977 | 34.2 | 4.27 | YES |
| lsb | 0.9995 | 41.1 | 2.00 | NO |
| dct | 0.9985 | 36.1 | 3.16 | NO |
| cw | 1.0000 | inf | 0.00 | NO |
| deepfool | 1.0000 | 51.1 | 0.50 | YES |
| auto_pgd | 0.9943 | 30.1 | 7.89 | YES |
| patch | 0.9999 | 48.3 | 0.13 | NO |
| gaussian | 0.9997 | 42.5 | 1.49 | NO |
| jsma | 0.9999 | 50.1 | 0.08 | NO |
