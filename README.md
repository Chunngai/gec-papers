# gec-papers

## GEC
- [x] 2021/1/5 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](#bert)
- [x] 2021/1/5 [Incorporating BERT into Neural Machine Translation](#bert-nmt)
- [x] 2021/1/6 [Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction](#bert-gec)
- [x] 2021/1/6 [GECToR - Grammatical Error Correction: Tag, Not Rewrite](#gector)
- [x] 2021/1/7 [MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking](#maskgec)
- [x] 2021/1/7 [Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)](#minimal-supervision)
- [x] 2021/1/7 [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model](#bart-gec)
- [x] 2021/1/9 [Chinese Grammatical Correction Using BERT-based Pre-trained Model](#chinese-bert-gec)
- [x] 2021/1/10 [Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction](#efficiency)
- [x] 2021/1/10 [Heterogeneous Recycle Generation for Chinese Grammatical Correction](#heterogeneous)  
- [x] 2021/1/10 [TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020](#chinese-bert-init)
- [x] 2021/1/11 [Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction](#local-beam-search)

## GED
- [x] 2021/1/6 [基于数据增强和多任务特征学习的中文语法错误检测方法](#chinese-multi-task)
- [x] 2021/1/11 [Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis](#score-based)
- [x] 2021/1/11 [CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared](#bert-crf)
- [x] 2021/1/11 [Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis](#resnet-bert)

## TODOs
- [ ] Seq2Edits: Sequence Transduction Using Span-level Edit Operations

---

## Seq2Seq
1. <span id="bert-gec">[ACL-2020] Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction</span>  
Applied the BERT-fused model for GEC. The BERT is finetuned with MLM and GED to fix the inconsistent input distribution between the raw data for BERT training and the GEC data. Pseudo-data and R2L are also used for performance boosting.  
https://github.com/kanekomasahiro/bert-gec

2. <span id="bart-gec">[AACL-2020] Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model</span>  
Used BART for GEC and says that BART can be a baseline for GEC, which can reach high performance by simple finetuning with GEC data instead of pseudo-data pretraining.  
https://github.com/Katsumata420/generic-pretrained-GEC

3. <span id="chinese-bert-gec">[IJCNLP-2020] Chinese Grammatical Correction Using BERT-based Pre-trained Model  
Tries BERT-init (BERT-encoder in the papar) and BERT-fused for Chinese GEC. The Chinese GEC ver. of *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, even less techniques used.

4. <span id="chinese-bert-init">[AACL-2020] TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020  
Uses BERT-init as in *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, which is also the same as the BERT-encoder in *Chinese Grammatical Correction Using BERT-based Pre-trained Model*.

## Seq Labeling
1. <span id="gector">[ACL-2020] GECToR - Grammatical Error Correction: Tag, Not Rewrite</span>  
Used a BERT sequence tagger. Developed custom task-specific g-transformations such as CASE, MERGE and so on. Since each time a token in the source sentence can only map an edit, iterative correction may be required. A 3-stage training strategy is used: data-aug pretraining - finetuning on err data - finetuning on err and err-free data.  
https://github.com/grammarly/gector

2. <span id="minimal-supervision">[AAAI-2020] Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)</span>  
Divides the GEC task into two stages: error identification and error correction. The first stage is a sequence tagging (remain, substitution, ...) task and a BERT is used for the second stage (correction).   
(Not very clear about the method proposed by the paper.)

3. <span id="score-based">[GED] [AACL-2020] Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis</span>  
Uses BiLSTM-CRF for GED, whose input is features concat composed of output of BERT, POS, POS score and PMI score. The scores are incorporated using a gating mechanism to avoid losing partial-order relationships when embedding continuous feature items.  
(Not very clear about the features used and the purpose of the gating mechanism.)

4. <span id="bert-crf">[GED] [AACL-2020] CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared]</span>  
Uses BERT + CRF.

5. <span id="resnet-bert">[GED] [AACL-2020] Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis</span>  
Applies res on BERT for GED. The encoded hidden repr is added with the emd and fed into the output layer.  
(Also related to GEC but not detailed, thus catogorize as GED.)

## Combination
1. <span id="efficiency">[EMNLP-2020] Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction</span>  
Combines a sequence tagging model for erroneous span detection and a seq2seq model for erroneous span correction to make the GEC process more efficient. The sequence tagging model (BERT-like) looks for spans needing to be corrected by outputting binary vectors, and the seq2seq model receives inputs annotated according to the outputs of the sequence tagging model and only produces outputs corresponding to the detected spans. Pseudo-data is used for pre-training the ESD and ESC models.

2. <span id="heterogeneous">[COLING-2020] Heterogeneous Recycle Generation for Chinese Grammatical Correction</span>  
Makes use of a sequence editing model, a seq2seq model and a spell checker to correct different kinds of errors (small scale errors, large scale errors and spell errors respectively). Iterative decoding is applied on (sequence editing model, seq2seq model). The proposed method needs not data-aug but still achieves comparable performance.

## Multi-Task Learning
1. <span id="chinese-multi-task">[GED] [CCL-2020] 基于数据增强和多任务特征学习的中文语法错误检测方法</span>  
Implements Chinese GED through data-augmentation and pretrained BERT finetuned using multi-task learning. The data-augmentation method applied here is simple, including manipulations such as insertions, deletions and so on. Some rules are designed to maintain the meanings of sentences. The Chinese BERT is used for GED with a CRF layer on top. It is finetuned through multi-task learning: pos tagging, parsing and grammar error detection.

## Beam Search
1. <span id="local-beam-search"> [COLING-2020] Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction</span>  
Proposes a local beam search method to output diverse outputs. The proposed method generates more diverse outputs than the plain beam search, and only modifies where should be corrected rather than changing the whole sequence as the global beam search. The copy factor in the copy-augmented Transformer is used as a penalty score.

## Dynamic Masking
1. <span id="maskgec">[AAAI-2020] MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking</span>  
Proposed a dynamic masking method for data-augmentation and generalization boosting. In each epoch each sentence is introduced noises with a prob by some manipulations, including padding substitution, random substution, word frequency substitution and so on.

## Related
1. <span id="bert-nmt">[NMT] [ICLR-2020] Incorporating BERT into Neural Machine Translation</span>  
Proposed a BERT-fused model. Comparing with the Vanilla Transformer, the proposed model has additionally one BERT-Enc Attention module in the encoder and a BERT-Dec Attention module in the decoder. Both of the additional modules are for incorporating features extracted by BERT whose weights are fixed. A Vanilla Transformer is trained in the first training stage, and in the second stage the BERT and additional modules are trained together.  
https://github.com/bert-nmt/bert-nmt
