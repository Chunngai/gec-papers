# gec-papers

Papers read since 2021.

- [x] [基于数据增强和多任务特征学习的中文语法错误检测方法](#1)
- [x] [GECToR - Grammatical Error Correction: Tag, Not Rewrite](#2)
- [x] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](#3)
- [x] [Incorporating BERT into Neural Machine Translation](#4)
- [x] [Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction](#5)
- [x] [MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking](#6)
- [x] [Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)](#7)
- [x] [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model](#8)

- [ ] Chinese Grammatical Correction Using BERT-based Pre-trained Model
- [ ] Seq2Edits: Sequence Transduction Using Span-level Edit Operations
- [ ] Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction
- [ ] Heterogeneous Recycle Generation for Chinese Grammatical Correction

# Pretraining
1. <span id="1">[CCL-2020] 基于数据增强和多任务特征学习的中文语法错误**检测**方法</span>  
Implements Chinese GED through data-augmentation and pretrained BERT finetuned using multi-task learning. The data-augmentation method applied here is simple, including manipulations such as insertions, deletions and so on. Some rules are designed to maintain the meanings of sentences. The Chinese BERT is used for GED with a CRF layer on top. It is finetuned through multi-task learning: pos tagging, parsing and grammar error detection.

2. <span id="2">[ACL-2020] GECToR - Grammatical Error Correction: Tag, Not Rewrite</span>  
Used a BERT sequence tagger. Developed custom task-specific g-transformations such as CASE, MERGE and so on. Since each time a token in the source sentence can only map an edit, iterative correction may be required. A 3-stage training strategy is used: data-aug pretraining - finetuning on err data - finetuning on err and err-free data.  
https://github.com/grammarly/gector


3. <span id="5">[ACL-2020] Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction</span>  
Applied the BERT-fused model for GEC. The BERT is finetuned with MLM and GED to fix the inconsistent input distribution between the raw data for BERT training and the GEC data. Pseudo-data and R2L are also used for performance boosting.  
https://github.com/kanekomasahiro/bert-gec

4. <span id="7">[AAAI-2020] Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)</span>  
Divides the GEC task into two stages: error identification and error correction. The first stage is a sequence tagging (remain, substitution, ...) task and a BERT is used for the second stage (correction).   
(Not very clear about the method proposed by the paper.)

5. <span id="8">[IJNLP-2020] Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model</span>  
Used BART for GEC and says that BART can be a baseline for GEC, which can reach high performance by simple finetuning with GEC data instead of pseudo-data pretraining.  
https://github.com/Katsumata420/generic-pretrained-GEC

# Data-Augmentation
1. <span id="6">[AAAI-2020] MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking</span>  
Proposed a dynamic masking method for data-augmentation and generalization boosting. In each epoch each sentence is introduced noises with a prob by some manipulations, including padding substitution, random substution, word frequency substitution and so on.

# Related
1. <span id="4">[NMT] [ICLR-2020] Incorporating BERT into Neural Machine Translation</span>  
Proposed a BERT-fused model. Comparing with the Vanilla Transformer, the proposed model has additionally one BERT-Enc Attention module in the encoder and a BERT-Dec Attention module in the decoder. Both of the additional modules are for incorporating features extracted by BERT whose weights are fixed. A Vanilla Transformer is trained in the first training stage, and in the second stage the BERT and additional modules are trained together.  
https://github.com/bert-nmt/bert-nmt
