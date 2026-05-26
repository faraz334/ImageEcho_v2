# Engine Reference

All 10 ImageEcho attack engines explained.

---

## 1. FGSM — Fast Gradient Sign Method
**Speed:** ⚡⚡⚡⚡⚡  **Strength:** ★★★

One-step attack. Computes the gradient of the loss with respect
to the input and steps in the sign direction.
Best for: quick testing, baseline comparison.

---

## 2. PGD — Projected Gradient Descent
**Speed:** ⚡⚡⚡  **Strength:** ★★★★

Multi-step FGSM with random initialisation inside the epsilon-ball.
Projects back after each step. Considered the gold standard iterative attack.

Best for: strong white-box attacks.

---

## 3. MI-FGSM (LSB Engine)
**Speed:** ⚡⚡⚡⚡⚡  **Strength:** ★★

Least Significant Bit flipping. No gradient required.
Flips the N least significant bits of each pixel channel.

Best for: steganography-style invisible perturbations.

---

## 4. DCT — Discrete Cosine Transform
**Speed:** ⚡⚡⚡⚡⚡  **Strength:** ★★★

Injects noise into high-frequency DCT coefficients of 8x8 pixel blocks.
High-frequency changes are invisible to humans but disruptive to CNNs.

Best for: frequency-domain invisible attacks.

---

## 5. C&W — Carlini-Wagner L2
**Speed:** ⚡  **Strength:** ★★★★

Optimization-based attack using Adam optimizer in tanh-space.
Minimises L2 distance while maximising misclassification confidence.
Produces the tightest, most invisible perturbations of any method.

Best for: maximum invisibility, research quality results.

---

## 6. DeepFool
**Speed:** ⚡⚡⚡  **Strength:** ★★★

Iteratively finds the minimal perturbation needed to cross the decision
boundary using linearised approximation of the boundary geometry.

Best for: finding the theoretical minimum perturbation.

---

## 7. AutoPGD — Automatic PGD
**Speed:** ⚡⚡  **Strength:** ★★★★

PGD with adaptive step size scheduling. Halves the step when loss
plateaus and restarts from the best found point.

Best for: better convergence than fixed-step PGD.

---

## 8. Patch Attack
**Speed:** ⚡⚡⚡  **Strength:** ★★★★

Concentrates all perturbation budget inside one square region.
Outside the patch: completely clean. Very high local perturbation.

Best for: physical world attacks, patch printing.

---

## 9. Gaussian — Frequency-Weighted Noise
**Speed:** ⚡⚡⚡⚡⚡  **Strength:** ★★

Structured Gaussian noise weighted by a 2D DFT frequency mask.
Concentrates noise in high-frequency areas where humans are least sensitive.

Best for: fast invisible noise baseline.

---

## 10. JSMA — Jacobian Saliency Map Attack
**Speed:** ⚡⚡⚡⚡  **Strength:** ★★★

Computes per-pixel saliency using the model Jacobian, then perturbs
only the top-k most salient pixels maximally in the gradient direction.
Produces extremely sparse perturbations.

Best for: sparse targeted attacks, interpretability research.
