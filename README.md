# GAT-ETM
## Overview
GAT-ETM is an end-to-end graph-based embedded topic model that jointly learning a knowledge graph of medical codes (in our model, ICD and ATC; but it is replaceable) and patients EHRs. Our model is able to learn multimodal (disease ICD and drug ATC) topics based on co-occurence patterns from EHR and semantic similarity from knowledge graph. Our idea is to leverage existing knowledge to aid EHR modeling especially for rare medical codes. This repo is the code used in a journal paper "Modeling electronic health record data using an end-to-end knowledge-graph-informed topic model". Please refer to the paper  (https://www.nature.com/articles/s41598-022-22956-w) for details. 
![My Image](GAT-ETM.png)

## Preparing Data:
**EHR data**

**Pre-trained code embeddings**

## Training

## Loading Trained Model and Evalutating

## Citation: 
Please cite the following paper if you use this source in your work.
```bibtex
@article{zou2022modeling,
  title={Modeling electronic health record data using an end-to-end knowledge-graph-informed topic model},
  author={Zou, Yuesong and Pesaranghader, Ahmad and Song, Ziyang and Verma, Aman and Buckeridge, David L and Li, Yue},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={17868},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```
For any clarification, comments, or suggestions please create an issue or contact \url{https://www.cs.mcgill.ca/~yueli/}{Dr. Yue Li} and \url{https://zouyuesong.github.io/}{Yuesong Zou}.
