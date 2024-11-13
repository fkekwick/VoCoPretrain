---
license: mit
---
Dataset for CVPR 2024 paper, [**"VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis"**](https://arxiv.org/abs/2402.17300)

Authors: Linshan Wu, <a href="https://scholar.google.com/citations?user=PfM5gucAAAAJ&hl=en">Jiaxin Zhuang</a>, and <a href="https://scholar.google.com/citations?hl=en&user=Z_t5DjwAAAAJ">Hao Chen</a>

## Download Dataset
```
cd VoCo
mkdir data
huggingface-cli download Luffy503/VoCo-10k  --repo-type dataset --local-dir .  --cache-dir ./cache
```