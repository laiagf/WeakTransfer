# WeakTransfer

This is the repository for the paper **The more the merrier: The effects of data volume and transfer learning on creating generalisable deep learning PAM detectors**

This study investigated the performance of different transferability strategies (cross-dataset deployment, single-source limited data retraining, random fine-tuning and active fine-tuning) for a TCN-based detector trained to classify 4-minute acoustic recordings based on the presence/absence of sperm whale clicks. The original detector can be found @ github.com/laiagf/WeakDetector. This repository includes code for fine-tuning the model via random and active sampling.


## Structure

`transfer/` includes all the necessary code for loading and fine-tuning pretrained models

`files/` contains CSV files with annotations for training and evaluation for the datasets used in our paper (same data as WeakDetector). Acompanying audio data is available on demand.

`experiments/` contains a practical implementation of the transfer module:

    *  `random_finetune/` contains code to systematically fine-tune with random sampling trained models to new data sources using different target dataset sizes
    
    *  `active_finetune/` contains code to systematically fine-tune with active sampling trained models to new data sources using different target dataset sizes, giving 4 different options for active selection: standard uncertainty sampling, weighted uncertainty sampling, class-balanced uncertainty sampling, and class-balanced weighted uncertainty sampling

## How to fine-tune your own models?

Tutorial coming very soon

## Paper abstract
Passive acoustic monitoring (PAM) generates large datasets that are costly to analyse manually, creating the need for accurate automated detection methods. While deep learning (DL) has shown promise for PAM-related tasks, the transferability of trained models to unseen datasets remains a critical challenge. To address this, we evaluated four methodological approaches for maximising the transferability of a DL sperm whale (Physeter macrocephalus) click detector: (1) cross-environment evaluation (training on five datasets, testing on an unseen sixth), (2) limited target-domain data training (model retraining using only a small, site-specific dataset), (3) pretraining with random fine-tuning , and (4) pretraining with active fine-tuning (selecting fine-tuning samples based on model uncertainty).  While trained models saw a drop in performance when tested on a new dataset, those pretrained on large diverse datasets outperformed limited target-specific training even when deployment conditions were not represented during training, confirming training diversity improves transferability. Fine-tuning with 500 target recordings effectively mitigated performance drops when deploying a trained model to new datasets. Active fine-tuning consistently outperformed all other approaches, although the improvement over random fine-tuning was marginal. For new studies, we recommend using models pretrained on diverse datasets and fine-tuning them with minimal target data. 
