# dcjdan-fault-diagnosis

Hi!

This repository contains a PyTorch implementation of the **DCJDAN** model from our paper:

**"Deep Coupled Joint Distribution Adaptation Network: A Method for Intelligent Fault Diagnosis Between Artificial and Real Damages"**  
📄 IEEE Transactions on Instrumentation and Measurement, 2020  
🔗 [Paper Link](https://doi.org/10.1109/TIM.2020.3043510)


## Research question

In real-world industrial applications, fault diagnosis models are often trained on data from artificially induced damage, which significantly differs from naturally occurring faults. This domain gap limits model performance in deployment.
This project proposes DCJDAN, a dual-stream deep adaptation network that aligns feature distributions and improves generalization from lab conditions to real conditions.


## 🌟 Highlights
- Two-stream untied CNNs to model domain-specific representations.
- Joint Distribution Adaptation (JDA) for reducing domain gaps.
- Supports diagnosis from artificial → real damage domains (TIM & TDM tasks).
- Outperforms state-of-the-art models by 10%–40% in benchmark experiments.


## 📦 Dataset
- Paderborn Bearing Dataset  
- CWRU Dataset  
- IMS Dataset  
📥 Instructions to download and preprocess: see [`datasets/README.md`](datasets/README.md)

## 🚀 Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/dcjdan-fault-diagnosis.git
cd dcjdan-fault-diagnosis
pip install -r requirements.txt
python src/train.py --task TIM_A

## 🧠 Citation
bibtex
@article{tan2020dcjdan,
  title={Deep Coupled Joint Distribution Adaptation Network: A Method for Intelligent Fault Diagnosis Between Artificial and Real Damages},
  author={Tan, Yongwen and Guo, Liang and Gao, Hongli and Zhang, Li},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2020},
  doi={10.1109/TIM.2020.3043510}
}

## 📬 Contact
For questions or collaborations: yongwentan12@gmail.com








