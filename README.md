# gec-papers

## Update Notes
**2022/7/28: {D}: Papers of GEC/GED datasets. {LOTE}: Papers of GEC/GED for languages other than English.**

**2022/5/18: Updating. The papers will be organized by publication years. Note that the key words are not from the paper authors, they are created by the repo author.**

## GEC Papers 2022
1. **Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction**
+ Authors: Maksym Tarnavskyi, Artem Chernodub, Kostiantyn Omelianchuk
+ Conference: ACL
+ Link: https://aclanthology.org/2022.acl-long.266/
+ Code: https://github.com/MaksTarnavskyi/gector-large
+ <details>
    <summary>Abstract</summary>
    In this paper, we investigate improvements to the GEC sequence tagging architecture with a focus on ensembling of recent cutting-edge Transformer-based encoders in Large configurations. We encourage ensembling models by majority votes on span-level edits because this approach is tolerant to the model architecture and vocabulary size. Our best ensemble achieves a new SOTA result with an F0.5 score of 76.05 on BEA-2019 (test), even without pre-training on synthetic datasets. In addition, we perform knowledge distillation with a trained ensemble to generate new synthetic training datasets, “Troy-Blogs” and “Troy-1BW”. Our best single sequence tagging model that is pretrained on the generated Troy- datasets in combination with the publicly available synthetic PIE dataset achieves a near-SOTA result with an F0.5 score of 73.21 on BEA-2019 (test). The code, datasets, and trained models are publicly available.
  </details>

2. **Interpretability for Language Learners Using Example-Based Grammatical Error Correction**
+ Authors: Masahiro Kaneko, Sho Takase, Ayana Niwa, Naoaki Okazaki
+ Conference: ACL
+ Link: https://aclanthology.org/2022.acl-long.496/
+ Code: https://github.com/kanekomasahiro/eb-gec
+ <details>
    <summary>Abstract</summary>
    Grammatical Error Correction (GEC) should not focus only on high accuracy of corrections but also on interpretability for language learning.However, existing neural-based GEC models mainly aim at improving accuracy, and their interpretability has not been explored.A promising approach for improving interpretability is an example-based method, which uses similar retrieved examples to generate corrections. In addition, examples are beneficial in language learning, helping learners understand the basis of grammatically incorrect/correct texts and improve their confidence in writing.Therefore, we hypothesize that incorporating an example-based method into GEC can improve interpretability as well as support language learners.In this study, we introduce an Example-Based GEC (EB-GEC) that presents examples to language learners as a basis for a correction result.The examples consist of pairs of correct and incorrect sentences similar to a given input and its predicted correction.Experiments demonstrate that the examples presented by EB-GEC help language learners decide to accept or refuse suggestions from the GEC output.Furthermore, the experiments also show that retrieved examples improve the accuracy of corrections.
  </details>
+ Key Words: Interpretability; kNN-MT; Seq2Seq; Application Oriented

3. **Adjusting the Precision-Recall Trade-Off with Align-and-Predict Decoding for Grammatical Error Correction**
+ Authors: Xin Sun, Houfeng Wang
+ Conference: ACL
+ Link: https://aclanthology.org/2022.acl-short.77/
+ Code: https://github.com/AutoTemp/Align-and-Predict
+ <details>
    <summary>Abstract</summary>
    Modern writing assistance applications are always equipped with a Grammatical Error Correction (GEC) model to correct errors in user-entered sentences. Different scenarios have varying requirements for correction behavior, e.g., performing more precise corrections (high precision) or providing more candidates for users (high recall). However, previous works adjust such trade-off only for sequence labeling approaches. In this paper, we propose a simple yet effective counterpart – Align-and-Predict Decoding (APD) for the most popular sequence-to-sequence models to offer more flexibility for the precision-recall trade-off. During inference, APD aligns the already generated sequence with input and adjusts scores of the following tokens. Experiments in both English and Chinese GEC benchmarks show that our approach not only adapts a single model to precision-oriented and recall-oriented inference, but also maximizes its potential to achieve state-of-the-art results. Our code is available at https://github.com/AutoTemp/Align-and-Predict.
  </details>
+ Key Words: Precision-Recall Trade-Off; Beam Search; Seq2Seq; Application Oriented

4. **“Is Whole Word Masking Always Better for Chinese BERT?”: Probing on Chinese Grammatical Error Correction**
+ Authors: Yong Dai, Linyang Li, Cong Zhou, Zhangyin Feng, Enbo Zhao, Xipeng Qiu, Piji Li, Duyu Tang
+ Conference: ACL Findings
+ Link: https://aclanthology.org/2022.findings-acl.1/
+ <details>
    <summary>Abstract</summary>
    Whole word masking (WWM), which masks all subwords corresponding to a word at once, makes a better English BERT model. For the Chinese language, however, there is no subword because each token is an atomic character. The meaning of a word in Chinese is different in that a word is a compositional unit consisting of multiple characters. Such difference motivates us to investigate whether WWM leads to better context understanding ability for Chinese BERT. To achieve this, we introduce two probing tasks related to grammatical error correction and ask pretrained models to revise or insert tokens in a masked language modeling manner. We construct a dataset including labels for 19,075 tokens in 10,448 sentences. We train three Chinese BERT models with standard character-level masking (CLM), WWM, and a combination of CLM and WWM, respectively. Our major findings are as follows: First, when one character needs to be inserted or replaced, the model trained with CLM performs the best. Second, when more than one character needs to be handled, WWM is the key to better performance. Finally, when being fine-tuned on sentence-level downstream tasks, models trained with different masking strategies perform comparably.
  </details>

5. **Type-Driven Multi-Turn Corrections for Grammatical Error Correction**
+ Authors: Shaopeng Lai, Qingyu Zhou, Jiali Zeng, Zhongli Li, Chao Li, Yunbo Cao, Jinsong Su
+ Conference: ACL Findings
+ Link: https://aclanthology.org/2022.findings-acl.254/
+ Code: https://github.com/DeepLearnXMU/TMTC
+ <details>
    <summary>Abstract</summary>
    Grammatical Error Correction (GEC) aims to automatically detect and correct grammatical errors. In this aspect, dominant models are trained by one-iteration learning while performing multiple iterations of corrections during inference. Previous studies mainly focus on the data augmentation approach to combat the exposure bias, which suffers from two drawbacks.First, they simply mix additionally-constructed training instances and original ones to train models, which fails to help models be explicitly aware of the procedure of gradual corrections. Second, they ignore the interdependence between different types of corrections.In this paper, we propose a Type-Driven Multi-Turn Corrections approach for GEC. Using this approach, from each training instance, we additionally construct multiple training instances, each of which involves the correction of a specific type of errors. Then, we use these additionally-constructed training instances and the original one to train the model in turn.Experimental results and in-depth analysis show that our approach significantly benefits the model training. Particularly, our enhanced model achieves state-of-the-art single-model performance on English GEC benchmarks. We release our code at Github.
  </details>
+ Key Words: Iterative Correction; Edit Operation; Sequence Labeling

6. **Frustratingly Easy System Combination for Grammatical Error Correction**
+ Authors: Muhammad Qorib, Seung-Hoon Na, Hwee Tou Ng
+ Conference: NAACL
+ Link: https://aclanthology.org/2022.naacl-main.143/
+ Code: https://github.com/nusnlp/esc
+ <details>
    <summary>Abstract</summary>
    In this paper, we formulate system combination for grammatical error correction (GEC) as a simple machine learning task: binary classification. We demonstrate that with the right problem formulation, a simple logistic regression algorithm can be highly effective for combining GEC models. Our method successfully increases the F0.5 score from the highest base GEC system by 4.2 points on the CoNLL-2014 test set and 7.2 points on the BEA-2019 test set. Furthermore, our method outperforms the state of the art by 4.0 points on the BEA-2019 test set, 1.2 points on the CoNLL-2014 test set with original annotation, and 3.4 points on the CoNLL-2014 test set with alternative annotation. We also show that our system combination generates better corrections with higher F0.5 scores than the conventional ensemble.
  </details>
+ Key Words: Ensembling; Edit Type; Linear Regression; Application Oriented

7. **{D} ErAConD: Error Annotated Conversational Dialog Dataset for Grammatical Error Correction**
+ Authors: Xun Yuan, Derek Pham, Sam Davidson, Zhou Yu
+ Conference: NAACL
+ Link: https://aclanthology.org/2022.naacl-main.5/
+ Code: https://github.com/yuanxun-yx/eracond
+ <details>
    <summary>Abstract</summary>
    Currently available grammatical error correction (GEC) datasets are compiled using essays or other long-form text written by language learners, limiting the applicability of these datasets to other domains such as informal writing and conversational dialog. In this paper, we present a novel GEC dataset consisting of parallel original and corrected utterances drawn from open-domain chatbot conversations; this dataset is, to our knowledge, the first GEC dataset targeted to a human-machine conversational setting. We also present a detailed annotation scheme which ranks errors by perceived impact on comprehension, making our dataset more representative of real-world language learning applications. To demonstrate the utility of the dataset, we use our annotated data to fine-tune a state-of-the-art GEC model. Experimental results show the effectiveness of our data in improving GEC model performance in a conversational scenario.
  </details>

8. **{D} MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction**
+ Authors: Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, Min Zhang
+ Conference: NAACL
+ Link: https://aclanthology.org/2022.naacl-main.227/
+ Code: https://github.com/HillZhang1999/MuCGEC
+ <details>
    <summary>Abstract</summary>
    This paper presents MuCGEC, a multi-reference multi-source evaluation dataset for Chinese Grammatical Error Correction (CGEC), consisting of 7,063 sentences collected from three Chinese-as-a-Second-Language (CSL) learner sources. Each sentence is corrected by three annotators, and their corrections are carefully reviewed by a senior annotator, resulting in 2.3 references per sentence. We conduct experiments with two mainstream CGEC models, i.e., the sequence-to-sequence model and the sequence-to-edit model, both enhanced with large pretrained language models, achieving competitive benchmark performance on previous and our datasets. We also discuss CGEC evaluation methodologies, including the effect of multiple references and using a char-based metric. Our annotation guidelines, data, and code are available at https://github.com/HillZhang1999/MuCGEC.
  </details>

9. **{D,LOTE} Czech Grammar Error Correction with a Large and Diverse Corpus**
+ Authors: Jakub Náplava, Milan Straka, Jana Straková, Alexandr Rosen
+ Conference: TACL
+ Link: https://aclanthology.org/2022.tacl-1.26/
+ Code: https://github.com/ufal/errant_czech
+ <details>
    <summary>Abstract</summary>
    We introduce a large and diverse Czech corpus annotated for grammatical error correction (GEC) with the aim to contribute to the still scarce data resources in this domain for languages other than English. The Grammar Error Correction Corpus for Czech (GECCC) offers a variety of four domains, covering error distributions ranging from high error density essays written by non-native speakers, to website texts, where errors are expected to be much less common. We compare several Czech GEC systems, including several Transformer-based ones, setting a strong baseline to future research. Finally, we meta-evaluate common GEC metrics against human judgments on our data. We make the new Czech GEC corpus publicly available under the CC BY-SA 4.0 license at http://hdl.handle.net/11234/1-4639.
  </details>

## GED Papers 2022
1. **Improving Chinese Grammatical Error Detection via Data augmentation by Conditional Error Generation**
+ Authors: Tianchi Yue, Shulin Liu, Huihui Cai, Tao Yang, Shengkang Song, TingHao Yu
+ Conference: ACL Findings
+ Link: https://aclanthology.org/2022.findings-acl.233/
+ Code: https://github.com/tc-yue/DA_CGED
+ <details>
    <summary>Abstract</summary>
    Chinese Grammatical Error Detection(CGED) aims at detecting grammatical errors in Chinese texts. One of the main challenges for CGED is the lack of annotated data. To alleviate this problem, previous studies proposed various methods to automatically generate more training samples, which can be roughly categorized into rule-based methods and model-based methods. The rule-based methods construct erroneous sentences by directly introducing noises into original sentences. However, the introduced noises are usually context-independent, which are quite different from those made by humans. The model-based methods utilize generative models to imitate human errors. The generative model may bring too many changes to the original sentences and generate semantically ambiguous sentences, so it is difficult to detect grammatical errors in these generated sentences. In addition, generated sentences may be error-free and thus become noisy data. To handle these problems, we propose CNEG, a novel Conditional Non-Autoregressive Error Generation model for generating Chinese grammatical errors. Specifically, in order to generate a context-dependent error, we first mask a span in a correct text, then predict an erroneous span conditioned on both the masked text and the correct span. Furthermore, we filter out error-free spans by measuring their perplexities in the original sentences. Experimental results show that our proposed method achieves better performance than all compared data augmentation methods on the CGED-2018 and CGED-2020 benchmarks.
  </details>
+ Key Words: Generative CGED; BERT Masking; Conditional Error Generation 

2. **Exploring the Capacity of a Large-scale Masked Language Model to Recognize Grammatical Errors**
+ Authors: Ryo Nagata, Manabu Kimura, Kazuaki Hanawa
+ Conference: ACL Findings
+ Link: https://aclanthology.org/2022.findings-acl.324/
+ Code: https://github.com/tc-yue/DA_CGED
+ <details>
    <summary>Abstract</summary>
    Abstract
    In this paper, we explore the capacity of a language model-based method for grammatical error detection in detail. We first show that 5 to 10% of training data are enough for a BERT-based error detection method to achieve performance equivalent to what a non-language model-based method can achieve with the full training data; recall improves much faster with respect to training data size in the BERT-based method than in the non-language model method. This suggests that (i) the BERT-based method should have a good knowledge of the grammar required to recognize certain types of error and that (ii) it can transform the knowledge into error detection rules by fine-tuning with few training samples, which explains its high generalization ability in grammatical error detection. We further show with pseudo error data that it actually exhibits such nice properties in learning rules for recognizing various types of error. Finally, based on these findings, we discuss a cost-effective method for detecting grammatical errors with feedback comments explaining relevant grammatical rules to learners.
  </details>

## SGED Papers 2022
1. **On Assessing and Developing Spoken ’Grammatical Error Correction’ Systems**
+ Authors: Yiting Lu, Stefano Bannò, Mark Gales
+ Conference: BEA
+ Link: https://aclanthology.org/2022.bea-1.9/
+ <details>
    <summary>Abstract</summary>
    Spoken ‘grammatical error correction’ (SGEC) is an important process to provide feedback for second language learning. Due to a lack of end-to-end training data, SGEC is often implemented as a cascaded, modular system, consisting of speech recognition, disfluency removal, and grammatical error correction (GEC). This cascaded structure enables efficient use of training data for each module. It is, however, difficult to compare and evaluate the performance of individual modules as preceeding modules may introduce errors. For example the GEC module input depends on the output of non-native speech recognition and disfluency detection, both challenging tasks for learner data.This paper focuses on the assessment and development of SGEC systems. We first discuss metrics for evaluating SGEC, both individual modules and the overall system. The system-level metrics enable tuning for optimal system performance. A known issue in cascaded systems is error propagation between modules.To mitigate this problem semi-supervised approaches and self-distillation are investigated. Lastly, when SGEC system gets deployed it is important to give accurate feedback to users. Thus, we apply filtering to remove edits with low-confidence, aiming to improve overall feedback precision. The performance metrics are examined on a Linguaskill multi-level data set, which includes the original non-native speech, manual transcriptions and reference grammatical error corrections, to enable system analysis and development.
  </details>

## GEC
<!-- - [x] 2021/1/6 [Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction](#bert-gec) [ACL-2020] √
- [x] 2021/1/6 [GECToR - Grammatical Error Correction: Tag, Not Rewrite](#gector) [ACL-2020] √
- [x] 2021/1/7 [MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking](#maskgec) [AAAI-2020]
- [x] 2021/1/7 [Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)](#minimal-supervision) [AAAI-2020]
- [x] 2021/1/7 [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model](#bart-gec) [AACL-2020] √
- [x] 2021/1/9 [Chinese Grammatical Correction Using BERT-based Pre-trained Model](#chinese-bert-gec) [IJCNLP-2020]
- [x] 2021/1/10 [Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction](#efficiency) [EMNLP-2020]
- [x] 2021/1/10 [Heterogeneous Recycle Generation for Chinese Grammatical Correction](#heterogeneous) [COLING-2020] √
- [x] 2021/1/10 [TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020](#chinese-bert-init) [AACL-2020]
- [x] 2021/1/11 [Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction](#local-beam-search) [COLING-2020]
- [x] 2021/1/12 [Seq2Edits: Sequence Transduction Using Span-level Edit Operations](#seq2edits) [EMNLP-2020]
- [x] 2021/1/12 [Adversarial Grammatical Error Correction](#adversarial) [EMNLP-2020]
- [x] 2021/1/17 Pseudo-Bidirectional Decoding for Local Sequence Transduction [EMNLP-2020]
- [x] 2021/1/18 Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data [ACL-2019]
- [x] 2021/1/18 An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction [ACL-2019]
- [x] 2021/1/19 Parallel Iterative Edit Models for Local Sequence Transduction [EMNLP-2019]
- [x] 2021/1/19 Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data [NAACL-2019]
- [x] 2021/1/20 A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning [ACL-2020]
- [x] 2021/1/20 The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction [ACL-2019]
- [x] 2021/1/20 TMU Transformer System Using BERT for Re-ranking at BEA 2019 Grammatical Error Correction on Restricted Track [ACL-2019]
- [x] 2021/1/21 Noisy Channel for Low Resource Grammatical Error Correction [ACL-2019]
- [x] 2021/1/22 The BLCU System in the BEA 2019 Shared Task [ACL-2019]
- [x] 2021/1/22 The AIP-Tohoku System at the BEA-2019 Shared Task [ACL-2019]
- [x] 2021/1/22 CUNI System for the Building Educational Applications 2019 Shared Task: Grammatical Error Correction [ACL-2019] -->

| Index | Date | Paper | Conference | Code | Note |
| :-: | --- | --- | --- | --- | --- |
| 1* | 21/1/6 | Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction (Kaneko et al.) | ACL-2020 | [Code](https://github.com/kanekomasahiro/bert-gec) | [Note](#bert-gec) |
| 2* | 21/1/6 | GECToR - Grammatical Error Correction: Tag, Not Rewrite (Omelianchuk et al.) | ACL-2020 | [Code](https://github.com/grammarly/gector) | [Note](#gector) |
| 3* | 21/1/7 | MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking (Zhao and Wang) | AAAI-2020 |  | [Note](#maskgec) |
| 4 | 21/1/7 | Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract) (Li et al.) | AAAI-2020 |  | [Note](#minimal-supervision) |
| 5* | 21/1/7 | Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model (Katsumata and Komachi) | AACL-2020 | [Code](https://github.com/Katsumata420/generic-pretrained-GEC) | [Note](#bart-gec) |
| 6 | 21/1/9 | Chinese Grammatical Correction Using BERT-based Pre-trained Model (Wang et al.) | IJCNLP-2020 |  | [Note](#chinese-bert-gec) |
| 7* | 21/1/10 | Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction (Chen et al.) | EMNLP-2020 |  | [Note](#efficiency) |
| 8* | 21/1/10 | Heterogeneous Recycle Generation for Chinese Grammatical Correction (Hinson et al.) | COLING-2020 |  | [Note](#heterogeneous) |
| 9 | 21/1/10 | TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020 (Wang and Komachi) | AACL-2020 |  | [Note](#chinese-bert-init) |
| 10 | 21/1/11 | Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction (Hotate et al.) | COLING-2020 |  | [Note](#local-beam-search) |
| 11 | 21/1/12 | Seq2Edits: Sequence Transduction Using Span-level Edit Operations (Stahlberg and Kumar) | EMNLP-2020 | [Code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/transformer_seq2edits.py) | [Note](#seq2edits) |
| 12 | 21/1/12 | Adversarial Grammatical Error Correction (Raheja and Alikaniotis) | EMNLP-2020 |  | [Note](#adversarial) |
| 13* | 21/1/17 | Pseudo-Bidirectional Decoding for Local Sequence Transduction (Zhou et al.) | EMNLP-2020 |  |  |
| 14 | 21/1/18 | Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data (Grundkiewicz et al.) | ACL-2019 |  |  |
| 15 | 21/1/18 | An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction (Kiyono et al.) | ACL-2019 |  |  |
| 16 | 21/1/19 | Parallel Iterative Edit Models for Local Sequence Transduction (Awasthi et al.) | EMNLP-2019 |  |  |
| 17 | 21/1/19 | Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data (Zhao et al.) | NAACL-2019 |  |  |
| 18 | 21/1/20 | A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning (Choe et al.) | ACL-2020 |  |  |
| 19 | 21/1/20 | The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction (Alikaniotis and Raheja) | ACL-2019 |  |  |
| 20 | 21/1/20 | TMU Transformer System Using BERT for Re-ranking at BEA 2019 Grammatical Error Correction on Restricted Track (Kaneko et al.) | ACL-2019 |  |  |
| 21 | 21/1/21 | Noisy Channel for Low Resource Grammatical Error Correction (Flachs et al.) | ACL-2019 |  |  |
| 22 | 21/1/22 | The BLCU System in the BEA 2019 Shared Task (Yang et al.) | ACL-2019 |  |  |
| 23 | 21/1/22 | The AIP-Tohoku System at the BEA-2019 Shared Task (Asano et al.) | ACL-2019 |  |  |
| 24 | 21/1/22 | CUNI System for the Building Educational Applications 2019 Shared Task: Grammatical Error Correction (Náplava and Straka) | ACL-2019 |  |  |
| 25 | 21/1/27 | Cross-Sentence Grammatical Error Correction (Chollampatt et al.) | ACL-2019 |  |  |

## GED
<!-- - [x] 2021/1/6 [基于数据增强和多任务特征学习的中文语法错误检测方法](#chinese-multi-task) [CCL-2020] √
- [x] 2021/1/11 [Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis](#score-based) [AACL-2020]
- [x] 2021/1/11 [CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared](#bert-crf) [AACL-2020]
- [x] 2021/1/11 [Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis](#resnet-bert) [AACL-2020]
- [x] 2021/1/11 [Chinese Grammatical Errors Diagnosis System Based on BERT at NLPTEA-2020 CGED Shared Task](#bert-bilstm-crf-3gram-seq2seq) [AACL-2020]
- [x] 2021/1/11 [Chinese Grammatical Error Detection Based on BERT Model](#bert-finetuned) [AACL-2020]
- [x] 2021/1/21 Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection [CICLING-2019] -->

| Index | Date | Paper | Conference | Code | Note |
| :-: | --- | --- | --- | --- | --- |
| 1* | 21/1/6 | 基于数据增强和多任务特征学习的中文语法错误检测方法 (Xie et al.) | CCL-2020 |  | [Note](#chinese-multi-task) |
| 2 | 21/1/11 | Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis (Cao et al.) | AACL-2020 |  | [Note](#score-based) |
| 3 | 21/1/11 | CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared (Wu and Wang) | AACL-2020 |  | [Note](#bert-crf) |
| 4 | 21/1/11 | Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis (Wang et al.) | AACL-2020 |  | [Note](#resnet-bert) |
| 5 | 21/1/11 | Chinese Grammatical Errors Diagnosis System Based on BERT at NLPTEA-2020 CGED Shared Task (Zan et al.) | AACL-2020 |  | [Note](#bert-bilstm-crf-3gram-seq2seq) |
| 6 | 21/1/11 | Chinese Grammatical Error Detection Based on BERT Model (Cheng and Duan) | AACL-2020 |  | [Note](#bert-finetuned) |
| 7 | 21/1/21 | Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection (Kaneko et al.) | CICLING-2019 |  |  |

## DA
<!-- - [x] Improving Grammatical Error Correction with Machine Translation Pairs [EMNLP-2020]
- [x] A Self-Refinement Strategy for Noise Reduction in Grammatical Error Correction [EMNLP-2020] -->

| Index | Date | Paper | Conference | Code | Note |
| :-: | --- | --- | --- | --- | --- |
| 1 | 21/1/11 | A Self-Refinement Strategy for Noise Reduction in Grammatical Error Correction (Mita et al.) | EMNLP-2020 |  |  |

## Related
<!-- - [x] 2021/1/5 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- [x] 2021/1/5 [Incorporating BERT into Neural Machine Translation](#bert-nmt) [ICLR-2020] √
- [x] 2021/1/17 Agreement on Target-Bidirectional LSTMs for Sequence-to-Sequence Learning [AAAI-2016]
- [x] 2021/1/17 Agreement on Target-bidirectional Neural Machine Translation [NAACL-2016]
- [x] 2021/1/17 Edinburgh Neural Machine Translation Systems for WMT 16
- [x] 2021/1/22 LIMIT-BERT: Linguistic Informed Multi-Task BERT [EMNLP-2020]
- [x] 2021/1/23 Distilling Knowledge Learned in BERT for Text Generation [ACL-2020]
- [x] 2021/1/23 Towards Making the Most of BERT in Neural Machine Translation [AAAI-2020]
- [x] 2021/1/23 Acquiring Knowledge from Pre-Trained Model to Neural Machine Translation [AAAI-2020] -->

| Index | Date | Paper | Conference | Code | Note |
| :-: | --- | --- | --- | --- | --- |
| 1 | 21/1/5 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al.) | NAACL-2019 |  |  |
| 2* | 21/1/5 | Incorporating BERT into Neural Machine Translation (Zhu et al.) | ICLR-2020 | [Code](https://github.com/bert-nmt/bert-nmt) | [Note](#bert-nmt) |
| 3 | 21/1/17 | Agreement on Target-Bidirectional LSTMs for Sequence-to-Sequence Learning (Liu et al.) | AAAI-2016 |  |  |
| 4 | 21/1/17 | Agreement on Target-bidirectional Neural Machine Translation (Liu et al.) | NAACL-2016 |  |  |
| 5* | 21/1/17 | Edinburgh Neural Machine Translation Systems for WMT 16 (Sennrich et al.) | WMT-2016 |  |  |
| 6 | 21/1/22 | LIMIT-BERT: Linguistic Informed Multi-Task BERT (Zhou et al.) | EMNLP-2020 |  |  |
| 7 | 21/1/23 | Distilling Knowledge Learned in BERT for Text Generation (Chen et al.) | ACL-2020 |  |  |
| 8 | 21/1/23 | Towards Making the Most of BERT in Neural Machine Translation (Yang et al.) | AAAI-2020 |  |  |
| 9 | 21/1/23 | Acquiring Knowledge from Pre-Trained Model to Neural Machine Translation (Weng et al.) | AAAI-2020 |  |  |
| 10 | 21/1/26 | Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting (Zhou et al.) | - |  |  |

---

## Seq2Seq
1. <span id="bert-gec">[ACL-2020] Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction</span>  
Applied the BERT-fused model for GEC. The BERT is finetuned with MLM and GED to fix the inconsistent input distribution between the raw data for BERT training and the GEC data. Pseudo-data and R2L are also used for performance boosting.  
https://github.com/kanekomasahiro/bert-gec

2. <span id="chinese-bert-gec">[IJCNLP-2020] Chinese Grammatical Correction Using BERT-based Pre-trained Model  
Tries BERT-init (BERT-encoder in the papar) and BERT-fused for Chinese GEC. The Chinese GEC ver. of *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, even less techniques used.

3. <span id="bart-gec">[AACL-2020] Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model</span>  
Used BART for GEC and says that BART can be a baseline for GEC, which can reach high performance by simple finetuning with GEC data instead of pseudo-data pretraining.  
https://github.com/Katsumata420/generic-pretrained-GEC

4. <span id="efficiency">[EMNLP-2020] Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction</span>  
Combines a sequence tagging model for erroneous span detection and a seq2seq model for erroneous span correction to make the GEC process more efficient. The sequence tagging model (BERT-like) looks for spans needing to be corrected by outputting binary vectors, and the seq2seq model receives inputs annotated according to the outputs of the sequence tagging model and only produces outputs corresponding to the detected spans. Pseudo-data is used for pre-training the ESD and ESC models.

## Seq2Edits
1. <span id="seq2edits">[EMNLP-2020] Seq2Edits: Sequence Transduction Using Span-level Edit Operations</span>  
Proposes a method for tasks containing many overlaps such as GEC. Uses Transformer with the decoder modified. The model receives a source sentence and at each inference time-step outputs a 3-tuple  which corresponds to an edit operation (error tag, source span end position, replacement). The error tag provides clear explanation. The paper conducts experiments on 5 NLP tasks containing many overlaps. Experiments with and without pretraining are conducted.  
(Not very clear about the modified decoder.)  
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/transformer_seq2edits.py

## Seq Labeling
1. <span id="gector">[ACL-2020] GECToR - Grammatical Error Correction: Tag, Not Rewrite</span>  
Used a BERT sequence tagger. Developed custom task-specific g-transformations such as CASE, MERGE and so on. Since each time a token in the source sentence can only map an edit, iterative correction may be required. A 3-stage training strategy is used: data-aug pretraining - finetuning on err data - finetuning on err and err-free data.  
https://github.com/grammarly/gector

2. <span id="minimal-supervision">[AAAI-2020] Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)</span>  
Divides the GEC task into two stages: error identification and error correction. The first stage is a sequence tagging (remain, substitution, ...) task and a BERT is used for the second stage (correction).   
(Not very clear about the method proposed by the paper.)

## Pipeline
1. <span id="heterogeneous">[COLING-2020] Heterogeneous Recycle Generation for Chinese Grammatical Correction</span>  
Makes use of a sequence editing model, a seq2seq model and a spell checker to correct different kinds of errors (small scale errors, large scale errors and spell errors respectively). Iterative decoding is applied on (sequence editing model, seq2seq model). The proposed method needs not data-aug but still achieves comparable performance.

## Multi-Task Learning
1. <span id="chinese-multi-task">[GED] [CCL-2020] 基于数据增强和多任务特征学习的中文语法错误检测方法</span>  
Implements Chinese GED through data-augmentation and pretrained BERT finetuned using multi-task learning. The data-augmentation method applied here is simple, including manipulations such as insertions, deletions and so on. Some rules are designed to maintain the meanings of sentences. The Chinese BERT is used for GED with a CRF layer on top. It is finetuned through multi-task learning: pos tagging, parsing and grammar error detection.

## Beam Search
1. <span id="local-beam-search">[COLING-2020] Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction</span>  
Proposes a local beam search method to output diverse outputs. The proposed method generates more diverse outputs than the plain beam search, and only modifies where should be corrected rather than changing the whole sequence as the global beam search. The copy factor in the copy-augmented Transformer is used as a penalty score.

## Adversarial Training
1. <span id="adversarial">[EMNLP-2020] Adversarial Grammatical Error Correction</span>  
The first approach to use adversarial training for GEC. Uses a seq2seq model as the generator and a sentence-pair classification model for the discriminator. The discriminator basically acts as a novel evaluation method for evaluating the outputs generated by the generator, which directly models the task. No other technique such as data augmentation is used.  
(Not very clear about the adversarial training.)

## Dynamic Masking
1. <span id="maskgec">[AAAI-2020] MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking</span>  
Proposed a dynamic masking method for data-augmentation and generalization boosting. In each epoch each sentence is introduced noises with a prob by some manipulations, including padding substitution, random substution, word frequency substitution and so on.

## NLPTEA
1. <span id="chinese-bert-init">[AACL-2020] TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020  
Uses BERT-init as in *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, which is also the same as the BERT-encoder in *Chinese Grammatical Correction Using BERT-based Pre-trained Model*.

2. <span id="score-based">[GED] [AACL-2020] Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis</span>  
Uses BiLSTM-CRF for GED, whose input is features concat composed of output of BERT, POS, POS score and PMI score. The scores are incorporated using a gating mechanism to avoid losing partial-order relationships when embedding continuous feature items.  
(Not very clear about the features used and the purpose of the gating mechanism.)

3. <span id="bert-crf">[GED] [AACL-2020] CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared]</span>  
Uses BERT + CRF.

4. <span id="resnet-bert">[GED] [AACL-2020] Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis</span>  
Applies res on BERT for GED. The encoded hidden repr is added with the emd and fed into the output layer.  
(Also related to GEC but not detailed, thus catogorize as GED.)

5. <span id="bert-bilstm-crf-3gram-seq2seq">[GED] [AACL-2020] Chinese Grammatical Errors Diagnosis System Based on BERT at NLPTEA-2020 CGED Shared Task</span>  
Uses BERT-BiLSTM-CRF for GED. Uses a hybrid system containing a 3-gram and a seq2seq for GEC.

6. <span id="bert-finetuned">[GED] [AACL-2020] Chinese Grammatical Error Detection Based on BERT Model</span>  
Uses BERT finetuned on GEC datasets.

## Related
1. <span id="bert-nmt">[NMT] [ICLR-2020] Incorporating BERT into Neural Machine Translation</span>  
Proposed a BERT-fused model. Comparing with the Vanilla Transformer, the proposed model has additionally one BERT-Enc Attention module in the encoder and a BERT-Dec Attention module in the decoder. Both of the additional modules are for incorporating features extracted by BERT whose weights are fixed. A Vanilla Transformer is trained in the first training stage, and in the second stage the BERT and additional modules are trained together.  
https://github.com/bert-nmt/bert-nmt
