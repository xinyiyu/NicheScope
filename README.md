# NicheScope
NicheScope is a computational framework for identifying and characterizing cell niches from spatial transcriptomics data. It jointly models a target cell’s gene expression and its local multicellular neighborhood to uncover multicellular niches (MCNs) and their corresponding niche-regulated cell states (NRCSs). NicheScope is robust and scalable, enabling reproducible analysis of tissue organization and functional microenvironments across diverse biological contexts.

![image](https://github.com/xinyiyu/NicheScope/blob/main/nichescope_demo.jpg)

## Installation
Git clone the repository and install the package:
```
conda env create -f environment.yml
conda activate NicheScope
python setup.py develop
```

## Reproducibility
We provide source codes for reproducing the NicheScope analysis in the main text:
* [B cell MCN in LN (Xenium and OpenST)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_OpenST_LN_B.ipynb)
* [T cell MCN in LN (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_LN_T.ipynb)
* [Tumor cell MCN in lung cancer (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_lung_tumor.ipynb)
* [TLS and stromal cell MCN in lung cancer (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_lung_TLS_stromal.ipynb)

## Reference
Xinyi Yu, Xiaomeng Wan, Leqi Tian, Yuheng Chen, Yuyao Liu, Tianwei Yu, Can Yang, Jiashun Xiao. NicheScope: Identifying Multicellular Niches and Niche-Regulated Cell States in Spatial Transcriptomics. doi: https://doi.org/10.1101/2025.08.21.671426.
 
## Contact information
Please contact Xinyi Yu (xyyu98@gmail.com) and Dr. Jiashun Xiao (jxiaoae@connect.ust.hk) if any enquiry.
