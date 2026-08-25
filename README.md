
Introduction：
LGDC is an omnidirectional image quality assessment method that takes a complete ERP image as input and predicts its quality score.

Key Innovations:

GPEC: Compensates for ERP projection distortion using latitude-longitude information

MQCA: Multi-scale quality feature extraction

ACDP: Adaptively captures distortions in different spatial regions

CGDC: Fuses features from MQCA and ACDC


Quick Start：

Installation：

git clone https://github.com/liziyi1234/LGDC.git

cd LGDC

pip install torch torchvision einops timm scipy pandas pillow tqdm

Data Preparation

Training：
python train.py

Inference 

File Description:

LGDC.py:	        Main model (MQCA + ACDC + CGDC)

GPEC.py	:       Geometry compensation module

train.py:	      Training entry point

config.py:	      Configuration parameters

MyDataset.py:	  Data loader

utils.py :       Utility functions

cyclic_shift.py:	Data augmentation

Citation:

@article{yan2026lgdc,

  title={Viewport-Unaware Blind Omnidirectional Image Quality Assessment: Learning from Geometry-Deformed Content},
  
  author={Yan, Jiebin and Li, Ziyi and Wu, Kangcheng and Chen, Pengfei and Zuo, Yifan and Chen, Junjie and Fang, Yuming},
  
  year={2026}
}

# LGDC: Viewport-Unaware Blind Omnidirectional Image Quality Assessment

<p align="center">
  <b>Learning from Geometry-Deformed Content</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#framework">Framework</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-preparation">Data Preparation</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#citation">Citation</a>
</p>

---

## 📖 Overview

**LGDC** is a viewport-unaware blind omnidirectional image quality assessment framework that directly takes a complete equirectangular projection (ERP) image as input and predicts its perceptual quality score.

Unlike viewport-based approaches that require extracting multiple viewports, LGDC performs quality assessment directly on the complete ERP representation. The framework jointly models multi-scale quality contexts, distortion-sensitive spatial representations, and ERP-aware geometric priors.

<div align="center">

**ERP Image → Hierarchical Feature Extraction → MQCA + ACDP → CGDC → GPEC → Quality Score**

</div>

---

## ✨ Key Innovations

### 🌐 GPEC: Geometry Prior Embedded Compensation

GPEC explicitly incorporates spherical position and latitude-dependent geometric priors to compensate for the non-uniform spatial characteristics introduced by ERP projection.

- Encodes latitude and longitude information.
- Models latitude-dependent geometric characteristics.
- Performs geometry-guided channel and spatial feature enhancement.

### 🔍 MQCA: Multi-granular Quality Context Aggregation

MQCA extracts complementary quality information from multiple receptive fields.

- Captures multi-scale quality patterns through adaptive scale-aware aggregation.
- Preserves fine-grained local quality details.
- Incorporates global contextual information.

### 🎯 ACDP: Adaptive Content-driven Degradation Perception

ACDP adaptively captures distortion-sensitive spatial regions according to the input content.

- Dynamically generates input-conditioned convolution kernels.
- Models distortion patterns at multiple spatial scales.
- Adaptively fuses multi-scale spatial responses.

### 🔗 CGDC: Context Guided Degradation Computation

CGDC integrates the complementary representations produced by MQCA and ACDP.

- Combines multi-scale quality contexts with distortion-aware features.
- Uses large and small receptive fields for complementary feature extraction.
- Employs gated fusion to adaptively balance broad context and fine-grained details.

---

## 🏗 Framework

The overall architecture of LGDC is illustrated below.

<p align="center">
  <img src="figures/framework.png" width="90%">
</p>

The framework consists of four major components:

| Module | Description |
|:---|:---|
| **MQCA** | Extracts multi-granular quality contexts from hierarchical features |
| **ACDP** | Adaptively perceives distortion-sensitive spatial patterns |
| **CGDC** | Integrates quality-context and distortion-aware representations |
| **GPEC** | Incorporates ERP geometric priors for feature compensation |

---

## ⚙️ Installation




# LGDC: Viewport-Unaware Blind Omnidirectional Image Quality Assessment

<p align="center">
  <b>Learning from Geometry-Deformed Content</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#framework">Framework</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-preparation">Data Preparation</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-BOIQA-blue.svg">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg">
  <img src="https://img.shields.io/badge/Input-ERP-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

---

## 📖 Overview

**LGDC** is a viewport-unaware blind omnidirectional image quality assessment framework that directly takes a complete equirectangular projection (ERP) image as input and predicts its perceptual quality score.

Unlike viewport-aware approaches that require extracting and processing multiple viewports, LGDC performs quality assessment directly on the complete ERP representation. The proposed framework jointly models multi-scale quality contexts, distortion-sensitive spatial representations, and ERP-aware geometric priors for perceptual quality prediction.

The overall pipeline can be summarized as:

<p align="center">

**ERP Image → Hierarchical Feature Extraction → MQCA + ACDP → CGDC → GPEC → Quality Score**

</p>

---

## ✨ Key Components

### 🌐 GPEC: Geometry Prior Embedded Compensation

The **Geometry Prior Embedded Compensation (GPEC)** module explicitly incorporates spherical position and latitude-dependent geometric priors to enhance feature representations under the non-uniform spatial characteristics of ERP images.

- Encodes spherical latitude and longitude information.
- Models latitude-dependent geometric characteristics.
- Performs geometry-guided channel-wise and spatial feature enhancement.
- Integrates geometry-enhanced features through residual fusion.

---

### 🔍 MQCA: Multi-granular Quality Context Aggregation

The **Multi-granular Quality Context Aggregation (MQCA)** module extracts complementary quality information from multiple spatial scales.

- Captures scale-adaptive quality patterns using parallel dilated convolutions.
- Preserves fine-grained local quality details.
- Incorporates global contextual information.
- Adaptively fuses multi-granular quality representations.

---

### 🎯 ACDP: Adaptive Content-driven Degradation Perception

The **Adaptive Content-driven Degradation Perception (ACDP)** module adaptively captures distortion-sensitive spatial patterns according to the input content.

- Dynamically generates input-conditioned convolution kernels.
- Perceives distortion patterns at multiple spatial scales.
- Produces multi-scale spatial responses.
- Adaptively fuses spatial responses to emphasize distortion-sensitive regions.

---

### 🔗 CGDC: Context Guided Degradation Computation

The **Context Guided Degradation Computation (CGDC)** module integrates the complementary representations generated by MQCA and ACDP.

- Combines multi-granular quality contexts and distortion-aware representations.
- Employs large and small receptive fields for complementary feature extraction.
- Uses channel-wise gated fusion to adaptively balance broad context and fine-grained details.
- Aggregates hierarchical features across backbone stages.

---

## 🏗 Framework

The overall architecture of LGDC is illustrated below.

<p align="center">
  <img src="figures/framework.png" width="90%">
</p>

LGDC consists of four major components:

| Module | Full Name | Description |
|:---:|:---|:---|
| **MQCA** | Multi-granular Quality Context Aggregation | Extracts multi-scale, local, and global quality contexts |
| **ACDP** | Adaptive Content-driven Degradation Perception | Adaptively captures distortion-sensitive spatial patterns |
| **CGDC** | Context Guided Degradation Computation | Integrates quality-context and distortion-aware representations |
| **GPEC** | Geometry Prior Embedded Compensation | Incorporates ERP geometric priors for feature enhancement |

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/liziyi1234/LGDC.git
cd LGDC

### Clone the repository

```bash
git clone https://github.com/liziyi1234/LGDC.git
cd LGDC

---

---

##  📂 Data Preparation

Please organize your dataset according to the structure below.
LGDC/
├── datasets/
│   ├── Dataset1/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── Dataset2/
│       ├── images/
│       └── labels/
│
├── LGDC.py
├── GPEC.py
├── train.py
├── config.py
├── MyDataset.py
├── utils.py
└── cyclic_shift.py

---
