# NicheScope
NicheScope is a computational framework for identifying and characterizing cell niches from spatial transcriptomics data. It jointly models a target cell’s gene expression and its local multicellular neighborhood to uncover multicellular niches (MCNs) and their corresponding niche-regulated cell states (NRCSs). NicheScope is robust and scalable, enabling reproducible analysis of tissue organization and functional microenvironments across diverse biological contexts.

![image](https://github.com/xinyiyu/NicheScope/blob/main/nichescope_demo.jpg)

## System requirements

NicheScope is implemented in Python and has been tested on Python 3.9 and 3.10 environments.

Operating systems:
- Linux (tested)
- macOS (compatible)

The required Python dependencies and package versions are provided in `environment.yml`.

No specialized hardware (e.g., GPU) is required. A standard desktop computer is sufficient for running NicheScope on typical spatial transcriptomics datasets.


## Installation
Git clone the repository and install the package:
```
conda env create -f environment.yml
conda activate NicheScope
python setup.py develop
```
The installation typically takes less than 10 minutes on a standard desktop computer.

## Demo
We provide a [demo notebook](https://github.com/xinyiyu/NicheScope/blob/main/demo/demo.ipynb) to illustrate the typical NicheScope workflow, including  
1. Required input data structure  
2. Running the `nichescope` function for niche detection  
3. Understanding the output and interpreting detected niches with visualization 
The demo uses [Xenium lymph node crop 1](https://drive.google.com/file/d/1oVS0nxrhf2TGYc3f-HI4dvubIa3uC4E_/view?usp=sharing) as an example dataset and performs B-cell-associated MCN detection.

The expected output includes:
- inferred multicellular niches (MCNs)
- niche scores for target cells
- niche-associated genes
- niche-associated cell types
- spatial visualization of detected niches

The demo typically completes within several minutes on a standard desktop computer.

## Reproducibility
We provide source codes for reproducing the NicheScope analysis in the main text:
* [B cell MCN in LN (Xenium and OpenST)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_OpenST_LN_B.ipynb)
* [T cell MCN in LN (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_LN_T.ipynb)
* [Adjacent-section MCN consistency in NSCLC (CosMx)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/CosMx_NSCLC_consistency.ipynb)
* [Tumor cell MCN in lung cancer (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_lung_tumor.ipynb)
* [TLS and stromal cell MCN in lung cancer (Xenium)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/Xenium_lung_TLS_stromal.ipynb)
* [Multi-condition niche discovery in primary and metastatic HNSCC (OpenST)](https://github.com/xinyiyu/NicheScope/blob/main/notebooks/OpenST_HNSCC.ipynb)

Processed spatial transcriptomics datasets used for the NicheScope real data analyses, together with additional analysis notebooks, are available on [Zenodo](https://doi.org/10.5281/zenodo.16943037).

## Reference
Xinyi Yu, Xiaomeng Wan, Leqi Tian, Yuheng Chen, Yuyao Liu, Tianwei Yu, Can Yang, Jiashun Xiao. NicheScope: Identifying Multicellular Niches and Niche-Regulated Cell States in Spatial Transcriptomics. doi: https://doi.org/10.1101/2025.08.21.671426.
 
## Contact information
Please contact Xinyi Yu (xyyu98@gmail.com) and Dr. Jiashun Xiao (jxiaoae@connect.ust.hk) if any enquiry.
