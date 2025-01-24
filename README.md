# NVBitFI for PCAHyperspectralClassifier

This repository contains the code for conducting the analysis presented in the paper **Analysis and Mitigation of Soft-errors in GPU-accelerated Hyperspectral Image Classifiers**.

The HSI classifier tested in this work is [PCAHyperspectralClassifier](https://github.com/gigernau/PCAHyperspectralClassifier/tree/main).

The fault injection tool used is [NVBitFI](https://github.com/NVlabs/nvbitfi). Refer to Chapter 4 of [this](https://webthesis.biblio.polito.it/33880/) document for a detailed explanation for setting up NVBitFI to simulate transient fault injections on the HSI classifier.

The logs obtained during the experiments can be found at the following Zenodo repository:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14563028.svg)](https://doi.org/10.5281/zenodo.14563028)

Check the description on Zenodo to find the link to the complete results.

## Packages needed for running the jupyter notebooks

### *injections_analysis.ipynb* and *kernels_sdc_critical_analysis_final.ipynb*:

- numpy 2.0.2
- pandas 2.1.4
- scikit-learn 1.5.2

### *Profile_analysis.ipynb*:

- matplotlib 3.8.0
- numpy 1.26.4
- pandas 2.2.2
- scikit-learn 1.6.0
- seaborn 0.13.2
