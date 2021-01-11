# gec-papers

Papers read since 2021.

## GEC
- [x] [GECToR - Grammatical Error Correction: Tag, Not Rewrite](#2)
- [x] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](#3)
- [x] [Incorporating BERT into Neural Machine Translation](#4)
- [x] [Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction](#5)
- [x] [MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking](#6)
- [x] [Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)](#7)
- [x] [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model](#8)
- [x] [Chinese Grammatical Correction Using BERT-based Pre-trained Model](#9)
- [x] [Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction](#10)
- [x] [Heterogeneous Recycle Generation for Chinese Grammatical Correction](#11)  
- [x] [TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020](#12)

## GED
- [x] [基于数据增强和多任务特征学习的中文语法错误检测方法](#1)
- [x] [Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis](#13)
- [x] [CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared](#14)
- [x] [Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis](#15)

## TODOs
- [ ] Seq2Edits: Sequence Transduction Using Span-level Edit Operations

---

## Seq2Seq
1. <span id="5">[ACL-2020] Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction</span>  
Applied the BERT-fused model for GEC. The BERT is finetuned with MLM and GED to fix the inconsistent input distribution between the raw data for BERT training and the GEC data. Pseudo-data and R2L are also used for performance boosting.  
https://github.com/kanekomasahiro/bert-gec

2. <span id="8">[AACL-2020] Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model</span>  
Used BART for GEC and says that BART can be a baseline for GEC, which can reach high performance by simple finetuning with GEC data instead of pseudo-data pretraining.  
https://github.com/Katsumata420/generic-pretrained-GEC

3. <span id="9">[IJCNLP-2020] Chinese Grammatical Correction Using BERT-based Pre-trained Model  
Tries BERT-init (BERT-encoder in the papar) and BERT-fused for Chinese GEC. The Chinese GEC ver. of *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, even less techniques used.

4. <span id="12">[AACL-2020] TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020  
Uses BERT-init as in *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, which is also the same as the BERT-encoder in *Chinese Grammatical Correction Using BERT-based Pre-trained Model*.

## Seq Labeling
1. <span id="2">[ACL-2020] GECToR - Grammatical Error Correction: Tag, Not Rewrite</span>  
Used a BERT sequence tagger. Developed custom task-specific g-transformations such as CASE, MERGE and so on. Since each time a token in the source sentence can only map an edit, iterative correction may be required. A 3-stage training strategy is used: data-aug pretraining - finetuning on err data - finetuning on err and err-free data.  
https://github.com/grammarly/gector

2. <span id="7">[AAAI-2020] Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)</span>  
Divides the GEC task into two stages: error identification and error correction. The first stage is a sequence tagging (remain, substitution, ...) task and a BERT is used for the second stage (correction).   
(Not very clear about the method proposed by the paper.)

3. <span id="13">[GED] [AACL-2020] Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis</span>  
Uses BiLSTM-CRF for GED, whose input is features concat composed of output of BERT, POS, POS score and PMI score. The scores are incorporated using a gating mechanism to avoid losing partial-order relationships when embedding continuous feature items.  
(Not very clear about the features used and the purpose of the gating mechanism.)

4. <span id="14">[GED] [AACL-2020] CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared]</span>  
Uses BERT + CRF.

5. <span id="15">[GED] [AACL-2020] Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis</span>  
Applies res on BERT for GED. The encoded hidden repr is added with the emd and fed into the output layer.  
(Also related to GEC but not detailed, thus catogorize as GED.)

## Combinations
1. <span id="10">[EMNLP-2020] Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction</span>  
Combines a sequence tagging model for erroneous span detection and a seq2seq model for erroneous span correction to make the GEC process more efficient. The sequence tagging model (BERT-like) looks for spans needing to be corrected by outputting binary vectors, and the seq2seq model receives inputs annotated according to the outputs of the sequence tagging model and only produces outputs corresponding to the detected spans. Pseudo-data is used for pre-training the ESD and ESC models.

2. <span id="11">[COLING-2020] Heterogeneous Recycle Generation for Chinese Grammatical Correction</span>  
Makes use of a sequence editing model, a seq2seq model and a spell checker to correct different kinds of errors (small scale errors, large scale errors and spell errors respectively). Iterative decoding is applied on (sequence editing model, seq2seq model). The proposed method needs not data-aug but still achieves comparable performance.

## Multi-Task Learning
1. <span id="1">[GED] [CCL-2020] 基于数据增强和多任务特征学习的中文语法错误检测方法</span>  
Implements Chinese GED through data-augmentation and pretrained BERT finetuned using multi-task learning. The data-augmentation method applied here is simple, including manipulations such as insertions, deletions and so on. Some rules are designed to maintain the meanings of sentences. The Chinese BERT is used for GED with a CRF layer on top. It is finetuned through multi-task learning: pos tagging, parsing and grammar error detection.

## Data-Augmentation
1. <span id="6">[AAAI-2020] MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking</span>  
Proposed a dynamic masking method for data-augmentation and generalization boosting. In each epoch each sentence is introduced noises with a prob by some manipulations, including padding substitution, random substution, word frequency substitution and so on.

## Related
1. <span id="4">[NMT] [ICLR-2020] Incorporating BERT into Neural Machine Translation</span>  
Proposed a BERT-fused model. Comparing with the Vanilla Transformer, the proposed model has additionally one BERT-Enc Attention module in the encoder and a BERT-Dec Attention module in the decoder. Both of the additional modules are for incorporating features extracted by BERT whose weights are fixed. A Vanilla Transformer is trained in the first training stage, and in the second stage the BERT and additional modules are trained together.  
https://github.com/bert-nmt/bert-nmt
