# OD-RTE

**OD-RTE: A One-Stage Object Detection Framework for Relational Triple Extraction**

This repository contains all the code of the official implementation for the paper: **[OD-RTE: A One-Stage Object Detection Framework for Relational Triple Extraction](https://aclanthology.org/2023.acl-long.623.pdf).** The paper has been accepted to appear at **ACL 2023**.

## How to Run the Code

## Data Source



1. Download the datasets to folder './data'. (The data used in this project is obtained from the [TPlinker](https://github.com/131250208/TPlinker-joint-extraction) project. )
2. Download the pretrained [BERT weights](https://huggingface.co/bert-base-cased) to folder './pretrained_models/bert-base-cased'.
3. Set hyperparameters and run ```main.py```

## Cite
```
@inproceedings{ning2023od,
  title={OD-RTE: A One-Stage Object Detection Framework for Relational Triple Extraction},
  author={Ning, Jinzhong and Yang, Zhihao and Sun, Yuanyuan and Wang, Zhizheng and Lin, Hongfei},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={11120--11135},
  year={2023}
}
```

## Acknowledgements

We would like to express our gratitude to the [TPlinker](https://github.com/131250208/TPlinker-joint-extraction) and [OneRel](https://github.com/China-ChallengeHub/OneRel) project for providing some code snippets that were used in this project. Their contributions have greatly helped in the development and implementation of our code. We appreciate their efforts and the open-source community for fostering collaboration and knowledge sharing.

