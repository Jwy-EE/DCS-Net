<div align="center">

# 🚀 DCS-Net

<h3><span style="color:#4F46E5;">A Physical-aware Deep Network</span> for <span style="color:#059669;">Robust Modulation Classification</span> under <span style="color:#DC2626;">High-dynamic Doppler Scenarios</span></h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-RadioML%202016.10a-0EA5E9?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-AMC-7C3AED?style=for-the-badge" />
</p>

</div>

---

## 📌 Affiliations

- <span style="color:#2563EB;"><b>Huimeisun</b></span> — Suzhou University of Technology  
- <span style="color:#7C3AED;"><b>Ziwen Qin</b></span> — Suzhou University of Technology  
- <span style="color:#EA580C;"><b>Wangye Jiang</b></span> — Suzhou University of Technology  
- <span style="color:#059669;"><b>Haoming Yang</b></span> — Jinling Institute of Technology  
- <span style="color:#DC2626;"><b>Jingya Zhang</b></span> — Suzhou University of Technology  

---

## 🌟 Overview

This repository contains the official implementation of **DCS-Net**, a **physical-aware deep network** for robust automatic modulation classification in challenging **high-dynamic Doppler environments**.

> Unlike conventional AMC methods that mainly rely on generic deep features, **DCS-Net explicitly incorporates signal physical characteristics into the network design** to improve robustness under Doppler-induced distortions.

### ✨ Core Modeling Components

DCS-Net jointly models:

- <span style="color:#2563EB;"><b>Raw I/Q features</b></span>  
- <span style="color:#059669;"><b>Magnitude / envelope information</b></span>  
- <span style="color:#DC2626;"><b>Phase-difference-aware representations</b></span>  

These complementary branches are fused together to improve recognition performance in difficult Doppler degradation scenarios.

---

## 📂 Data

We conduct experiments on the **RadioML 2016.10a** dataset.

### Dataset Summary

| Dataset | Modulation Formats | Samples |
|:--|:--|:--:|
| **RML2016.10a** | 8 digital formats: `8PSK`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `16QAM`, `64QAM`, `QPSK` <br/> 3 analog formats: `AM-DSB`, `AM-SSB`, `WBFM` | `(2 × 128)` |

### 🔽 Download

The original dataset can be obtained from:

- **DeepSig Official Website**
- **DeepSig: AI-Native Wireless Communications**

---

## 🗂 Repository Structure

```text
DCS-Net/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ main.py
├─ dataset_process.py
├─ assets/
│  └─ confusion_metrics_18dB.png
├─ models/
│  └─ model.py
├─ data_loader/
│  └─ data_loader.py
└─ util/
   ├─ config.py
   ├─ early_stop.py
   ├─ logger.py
   ├─ training.py
   └─ utils.py
```

## 🛠 Data Preparation

This project expects the processed dataset file at:

```text
data/radioml/RML2016.10a_Aerospace_corrupted.pkl
```

If you only have the original dataset file, first generate the processed version by running:

```bash
python dataset_process.py
```

By default, the script reads:

```text
data/radioml/RML2016.10a_dict.pkl
```

and writes:

```text
data/radioml/RML2016.10a_Aerospace_corrupted.pkl
```

---

## 🚂 Training

Run training with:

```bash
python main.py
```

### Example: full-SNR training

```bash
python main.py --dataset 2016.10a --epochs 100 --batch_size 256 --lr 0.001 --target_snrs all
```

### Example: selected SNR values only

```bash
python main.py --target_snrs -20,-18,-16
```

### Example: resume from checkpoint

```bash
python main.py --resume checkpoint/2016.10a_.pkl
```

## 📊 Results

The repository includes the confusion matrix of **DCS-Net at 18 dB**:

<div align="center">
  <img src="assets/confusion_metrics_18dB.png" alt="Confusion Matrix at 18 dB" width="700"/>
</div>

---

## 📝 Notes

- The current implementation supports the **RadioML 2016.10a** dataset format used in this repository.
- Runtime-generated directories such as `data/`, `checkpoint/`, and `log/` are excluded in `.gitignore`.
- The training script automatically saves the best-performing model according to validation accuracy.

---

## 📄 License

This project is released under the terms of the license provided in the **LICENSE** file.

---

## 🙏 Acknowledgments

We thank the authors of the RadioML dataset and the open-source community for their valuable contributions.
