# Cover Letter — Biomedical Signal Processing and Control (Elsevier)

**Date:** November 17, 2026  
**To:** The Editor-in-Chief, *Biomedical Signal Processing and Control* (Elsevier)  
**Subject:** Submission of Original Research Manuscript: *"Hybrid Multimodal Deep Learning and Handcrafted Dermoscopic Feature Fusion with Lesion-Grouped Validation and Clinical Threshold Optimization for Multiclass Skin Cancer Classification"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript entitled **"Hybrid Multimodal Deep Learning and Handcrafted Dermoscopic Feature Fusion with Lesion-Grouped Validation and Clinical Threshold Optimization for Multiclass Skin Cancer Classification"** for consideration for publication in *Biomedical Signal Processing and Control*.

### Biomedical Relevance and Scientific Originality
Automated classification of dermatoscopic images on benchmark repositories such as HAM10000 (10,015 images across 7 diagnostic categories) is frequently compromised in the literature by two critical methodological flaws: (1) duplicate-lesion data leakage caused by image-level splitting without grouping by `lesion_id`, and (2) reporting mean One-vs-Rest (OvR) binary accuracy ($\sim 90\text{--}95\%$) as if it were exact multiclass accuracy under severe class imbalance (67% benign nevi).

In this study, we address these challenges through a leak-free, end-to-end biomedical image processing and deep learning framework:
1. **Strict Lesion-Grouped Partitioning & Metric Disambiguation:** We enforce zero `lesion_id` overlap across 70% training, 15% validation, and 15% hold-out test partitions, and provide the exact algebraic relationship linking global multiclass accuracy and mean OvR binary accuracy.
2. **Systematic Progression and Ablation from Handcrafted Semiotics to Hybrid Deep Learning:** We quantify the incremental contribution and collinearity trade-offs of morphological hair removal (DullRazor), chromatic (RGB/HSV), textural (GLCM/LBP), and morphological (Otsu + log-transformed Hu moments) descriptors in a Random Forest baseline before fusing the 30-dimensional ABCD vector with an ImageNet-pretrained **EfficientNetV2B0** backbone.
3. **Rigorous Comparative Benchmarking and Statistical Auditing:** Under identical partitions, Sparse Focal Loss ($\gamma=2.0$), uniform class sampling, and multiple random seeds, our hybrid EfficientNetV2B0 significantly outperforms both Random Forest ensembles and heavier CNN baselines (**ResNet50** and **ConvNeXt-Tiny**, $p < 0.001$ via McNemar and paired cluster bootstrap tests) while requiring $\sim 4\times$ fewer parameters ($6.10 \times 10^6$).
4. **Probabilistic Calibration, Clinical Threshold Tuning, and Explainability:** We explicitly distinguish probabilistic calibration (Multiclass Brier Score = 0.284, Expected Calibration Error = 6.8%) from validation-tuned per-class decision thresholds ($\theta_k$), and provide a comprehensive clinical audit of melanoma false negatives, Grad-CAM visual explanations, and photometric perturbation robustness.

### Alignment with the Scope of *Biomedical Signal Processing and Control*
This work directly addresses the core scope of *BSPC* by integrating classical biomedical image processing (morphological filtering, Telea inpainting, co-occurrence texture analysis, invariant moments) with compact multimodal deep learning and rigorous clinical decision theory for non-invasive dermatological diagnosis.

### Declarations
This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. The author declares no competing financial or personal interests. All data and reproducible source code are publicly accessible.

Thank you for your time and consideration of our manuscript. We look forward to hearing from you.

Sincerely,

**Alexandre Ramos da Silva**  
Postgraduate Researcher in Data Science and Artificial Intelligence Applied to Healthcare  
GitHub Repository: https://github.com/aramos197442/skin_cancer
