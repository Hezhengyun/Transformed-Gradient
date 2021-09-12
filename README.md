# Transformed-Gradient
Boosting Adversarial Attacks with Transformed Gradient
This is tensorflow implementation for paper "Boosting Adversarial Attacks with Transformed Gradient"

We propose a Transformed Gradient method (TG), which consists of three steps: orignial gradient accumulation, gradient amplification and gradient truncation. This method achieves higher attack success rate with less perturbations. Meanwhile, we introduce the Frechet Inception Distance (FID) (codes see here)and Learned Perceptual Image Patch Similarity (LPIPS)(codes see here) respectively to evaluate fidelity and perceived distance from the original image, which is more comprehensive than only using  $ L_\infty $ norm as evaluation metrics.
