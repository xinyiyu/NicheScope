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

## Demo
We provide a [demo notebook](https://github.com/xinyiyu/NicheScope/blob/main/demo/demo.ipynb) to illustrate the typical NicheScope workflow, including  
1. Required input data structure  
2. Running the `nichescope` function for niche detection  
3. Understanding the output and interpreting detected niches with visualization 
The demo uses [Xenium lymph node crop 1](https://drive.google.com/file/d/1oVS0nxrhf2TGYc3f-HI4dvubIa3uC4E_/view?usp=sharing) as an example dataset and performs B cell niche detection.

## Reproducibility
We provide source codes for reproducing the NicheScope analysis in the main text:
* [B cell MCN in LN (Xenium and OpenST)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_OpenST_LN_B.ipynb)
* [T cell MCN in LN (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_LN_T.ipynb)
* [Tumor cell MCN in lung cancer (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_lung_tumor.ipynb)
* [TLS and stromal cell MCN in lung cancer (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_lung_TLS_stromal.ipynb)
* [Multi-condition niche discovery in primary and metastatic HNSCC (OpenST)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/OpenST_HNSCC.ipynb)

## Reference
Xinyi Yu, Xiaomeng Wan, Leqi Tian, Yuheng Chen, Yuyao Liu, Tianwei Yu, Can Yang, Jiashun Xiao. NicheScope: Identifying Multicellular Niches and Niche-Regulated Cell States in Spatial Transcriptomics. doi: https://doi.org/10.1101/2025.08.21.671426.
 
## Contact information
Please contact Xinyi Yu (xyyu98@gmail.com) and Dr. Jiashun Xiao (jxiaoae@connect.ust.hk) if any enquiry.
