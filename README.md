# Papers of Grammatical Error Correction

## Introduction
This repository lists papers of **Grammatical Error Correction (GEC)** and those of related topics,
such as **Grammatical Error Detection (GED)** and **Spoken Grammatical Error Correction (SGEC)**.

## Update Notes

**2023/3/25:** Add papers of multilingual GEC.

**2023/3/22:** Add a topic: syntax-enhanced.

**2023/3/21:** 
1. Reorganized the repo.
2. Divide GEC papers into:
   + GEC methods
   + GEC datasets
   + GEC evaluation
3. Remove symbols: {D} and {E}.
4. Changing symbols {L:`lang_codes`} to [`lang_codes`]

**2022/11/9:** 
1. Added a symbol {L:`lang_codes`} to indicate GEC for languages other than English, or multilingual GEC. Deprecated {LOTE}.
2. Added a symbol {E} to indicate studies for evaluating GEC models.
3. Added a topic: multi-lingual GEC.

**2022/7/28:**
Added some symbols.  
+ {D}: Papers of GEC/GED datasets. 
+ {LOTE}: Papers of GEC/GED for languages other than English.

**2022/5/18:**
Updating. The papers will be organized by publication years.

## Papers in 2022
### GEC Methods
+ **Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction**
  + Authors: Maksym Tarnavskyi, Artem Chernodub, Kostiantyn Omelianchuk
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-long.266/
  + Code: https://github.com/MaksTarnavskyi/gector-large
  + <details>
    <summary>Abstract</summary>
    In this paper, we investigate improvements to the GEC sequence tagging architecture with a focus on ensembling of recent cutting-edge Transformer-based encoders in Large configurations. We encourage ensembling models by majority votes on span-level edits because this approach is tolerant to the model architecture and vocabulary size. Our best ensemble achieves a new SOTA result with an F0.5 score of 76.05 on BEA-2019 (test), even without pre-training on synthetic datasets. In addition, we perform knowledge distillation with a trained ensemble to generate new synthetic training datasets, “Troy-Blogs” and “Troy-1BW”. Our best single sequence tagging model that is pretrained on the generated Troy- datasets in combination with the publicly available synthetic PIE dataset achieves a near-SOTA result with an F0.5 score of 73.21 on BEA-2019 (test). The code, datasets, and trained models are publicly available.
  </details>
[//]: # (+ Key Words: Empirical Study; Bigger PLMs; Ensembling Comparison; Knowledge Distilling)

+ **Interpretability for Language Learners Using Example-Based Grammatical Error Correction**
  + Authors: Masahiro Kaneko, Sho Takase, Ayana Niwa, Naoaki Okazaki
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-long.496/
  + Code: https://github.com/kanekomasahiro/eb-gec
  + <details>
    <summary>Abstract</summary>
    Grammatical Error Correction (GEC) should not focus only on high accuracy of corrections but also on interpretability for language learning.However, existing neural-based GEC models mainly aim at improving accuracy, and their interpretability has not been explored.A promising approach for improving interpretability is an example-based method, which uses similar retrieved examples to generate corrections. In addition, examples are beneficial in language learning, helping learners understand the basis of grammatically incorrect/correct texts and improve their confidence in writing.Therefore, we hypothesize that incorporating an example-based method into GEC can improve interpretability as well as support language learners.In this study, we introduce an Example-Based GEC (EB-GEC) that presents examples to language learners as a basis for a correction result.The examples consist of pairs of correct and incorrect sentences similar to a given input and its predicted correction.Experiments demonstrate that the examples presented by EB-GEC help language learners decide to accept or refuse suggestions from the GEC output.Furthermore, the experiments also show that retrieved examples improve the accuracy of corrections.
  </details>
[//]: # (+ Key Words: Interpretability; kNN-MT; Seq2Seq; Application Oriented)

+ **Adjusting the Precision-Recall Trade-Off with Align-and-Predict Decoding for Grammatical Error Correction**
  + Authors: Xin Sun, Houfeng Wang
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-short.77/
  + Code: https://github.com/AutoTemp/Align-and-Predict
  + <details>
      <summary>Abstract</summary>
      Modern writing assistance applications are always equipped with a Grammatical Error Correction (GEC) model to correct errors in user-entered sentences. Different scenarios have varying requirements for correction behavior, e.g., performing more precise corrections (high precision) or providing more candidates for users (high recall). However, previous works adjust such trade-off only for sequence labeling approaches. In this paper, we propose a simple yet effective counterpart – Align-and-Predict Decoding (APD) for the most popular sequence-to-sequence models to offer more flexibility for the precision-recall trade-off. During inference, APD aligns the already generated sequence with input and adjusts scores of the following tokens. Experiments in both English and Chinese GEC benchmarks show that our approach not only adapts a single model to precision-oriented and recall-oriented inference, but also maximizes its potential to achieve state-of-the-art results. Our code is available at https://github.com/AutoTemp/Align-and-Predict.
    </details>
[//]: # (+ Key Words: Precision-Recall Trade-Off; Beam Search; Seq2Seq; Application Oriented)

+ **[`zh`] “Is Whole Word Masking Always Better for Chinese BERT?”: Probing on Chinese Grammatical Error Correction**
  + Authors: Yong Dai, Linyang Li, Cong Zhou, Zhangyin Feng, Enbo Zhao, Xipeng Qiu, Piji Li, Duyu Tang
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2022.findings-acl.1/
  + <details>
      <summary>Abstract</summary>
      Whole word masking (WWM), which masks all subwords corresponding to a word at once, makes a better English BERT model. For the Chinese language, however, there is no subword because each token is an atomic character. The meaning of a word in Chinese is different in that a word is a compositional unit consisting of multiple characters. Such difference motivates us to investigate whether WWM leads to better context understanding ability for Chinese BERT. To achieve this, we introduce two probing tasks related to grammatical error correction and ask pretrained models to revise or insert tokens in a masked language modeling manner. We construct a dataset including labels for 19,075 tokens in 10,448 sentences. We train three Chinese BERT models with standard character-level masking (CLM), WWM, and a combination of CLM and WWM, respectively. Our major findings are as follows: First, when one character needs to be inserted or replaced, the model trained with CLM performs the best. Second, when more than one character needs to be handled, WWM is the key to better performance. Finally, when being fine-tuned on sentence-level downstream tasks, models trained with different masking strategies perform comparably.
    </details>

+ **Type-Driven Multi-Turn Corrections for Grammatical Error Correction**
  + Authors: Shaopeng Lai, Qingyu Zhou, Jiali Zeng, Zhongli Li, Chao Li, Yunbo Cao, Jinsong Su
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2022.findings-acl.254/
  + Code: https://github.com/DeepLearnXMU/TMTC
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) aims to automatically detect and correct grammatical errors. In this aspect, dominant models are trained by one-iteration learning while performing multiple iterations of corrections during inference. Previous studies mainly focus on the data augmentation approach to combat the exposure bias, which suffers from two drawbacks.First, they simply mix additionally-constructed training instances and original ones to train models, which fails to help models be explicitly aware of the procedure of gradual corrections. Second, they ignore the interdependence between different types of corrections.In this paper, we propose a Type-Driven Multi-Turn Corrections approach for GEC. Using this approach, from each training instance, we additionally construct multiple training instances, each of which involves the correction of a specific type of errors. Then, we use these additionally-constructed training instances and the original one to train the model in turn.Experimental results and in-depth analysis show that our approach significantly benefits the model training. Particularly, our enhanced model achieves state-of-the-art single-model performance on English GEC benchmarks. We release our code at Github.
    </details>
[//]: # (+ Key Words: Iterative Correction; Edit Operation; Sequence Labeling)

+ <a name="zhang-et-al-emnlp2022"></a>**[`en,zh`] SynGEC: Syntax-Enhanced Grammatical Error Correction with a Tailored GEC-Oriented Parser**
  + Authors: Yue Zhang, Bo Zhang, Zhenghua Li, Zuyi Bao, Chen Li, Min Zhang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.162/
  + Code: https://github.com/HillZhang1999/SynGEC
  + <details>
    <summary>Abstract</summary>
    This work proposes a syntax-enhanced grammatical error correction (GEC) approach named SynGEC that effectively incorporates dependency syntactic information into the encoder part of GEC models. The key challenge for this idea is that off-the-shelf parsers are unreliable when processing ungrammatical sentences. To confront this challenge, we propose to build a tailored GEC-oriented parser (GOPar) using parallel GEC training data as a pivot. First, we design an extended syntax representation scheme that allows us to represent both grammatical errors and syntax in a unified tree structure. Then, we obtain parse trees of the source incorrect sentences by projecting trees of the target correct sentences. Finally, we train GOPar with such projected trees. For GEC, we employ the graph convolution network to encode source-side syntactic information produced by GOPar, and fuse them with the outputs of the Transformer encoder. Experiments on mainstream English and Chinese GEC datasets show that our proposed SynGEC approach consistently and substantially outperforms strong baselines and achieves competitive performance. Our code and data are all publicly available at https://github.com/HillZhang1999/SynGEC.
  </details>

+ **[`en,ru`] Improved grammatical error correction by ranking elementary edits**
  + Authors: Alexey Sorokin
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.785/
  + Code: https://github.com/AlexeySorokin/
  + <details>
    <summary>Abstract</summary>
    We offer a two-stage reranking method for grammatical error correction: the first model serves as edit generator, while the second classifies the proposed edits as correct or false. We show how to use both encoder-decoder and sequence labeling models for the first step of our pipeline. We achieve state-of-the-art quality on BEA 2019 English dataset even using weak BERT-GEC edit generator. Combining our roberta-base scorer with state-of-the-art GECToR edit generator, we surpass GECToR by 2-3%. With a larger model we establish a new SOTA on BEA development and test sets. Our model also sets a new SOTA on Russian, despite using smaller models and less data than the previous approaches.
  </details>

+ **[`zh`] Linguistic Rules-Based Corpus Generation for Native Chinese Grammatical Error Correction**
  + Authors: Shirong Ma, Yinghui Li, Rongyi Sun, Qingyu Zhou, Shulin Huang, Ding Zhang, Li Yangning, Ruiyang Liu, Zhongli Li, Yunbo Cao, Haitao Zheng, Ying Shen
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.40/
  + Code: https://github.com/masr2000/CLG-CGEC
  + <details>
    <summary>Abstract</summary>
    Chinese Grammatical Error Correction (CGEC) is both a challenging NLP task and a common application in human daily life. Recently, many data-driven approaches are proposed for the development of CGEC research. However, there are two major limitations in the CGEC field: First, the lack of high-quality annotated training corpora prevents the performance of existing CGEC models from being significantly improved. Second, the grammatical errors in widely used test sets are not made by native Chinese speakers, resulting in a significant gap between the CGEC models and the real application. In this paper, we propose a linguistic rules-based approach to construct large-scale CGEC training corpora with automatically generated grammatical errors. Additionally, we present a challenging CGEC benchmark derived entirely from errors made by native Chinese speakers in real-world scenarios. Extensive experiments and detailed analyses not only demonstrate that the training data constructed by our method effectively improves the performance of CGEC models, but also reflect that our benchmark is an excellent resource for further development of the CGEC field.
  </details>

+ **[`zh`] From Spelling to Grammar: A New Framework for Chinese Grammatical Error Correction**
  + Authors: Xiuyu Wu, Yunfang Wu
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.63/
  + <details>
    <summary>Abstract</summary>
    Chinese Grammatical Error Correction (CGEC) aims to generate a correct sentence from an erroneous sequence, where different kinds of errors are mixed. This paper divides the CGEC task into two steps, namely spelling error correction and grammatical error correction. We firstly propose a novel zero-shot approach for spelling error correction, which is simple but effective, obtaining a high precision to avoid error accumulation of the pipeline structure. To handle grammatical error correction, we design part-of-speech (POS) features and semantic class features to enhance the neural network model, and propose an auxiliary task to predict the POS sequence of the target sentence. Our proposed framework achieves a 42.11 F-0.5 score on CGEC dataset without using any synthetic data or data augmentation methods, which outperforms the previous state-of-the-art by a wide margin of 1.30 points. Moreover, our model produces meaningful POS representations that capture different POS words and convey reasonable POS transition rules.
  </details>

+ **[`en,zh`] Sequence-to-Action: Grammatical Error Correction with Action Guided Sequence Generation**
  + Authors: Jiquan Li, Junliang Guo, Yongxin Zhu, Xin Sheng, Deqiang Jiang, Bo Ren, Linli Xu
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/21345
  + <details>
    <summary>Abstract</summary>
    The task of Grammatical Error Correction (GEC) has received remarkable attention with wide applications in Natural Language Processing (NLP) in recent years. While one of the key principles of GEC is to keep the correct parts unchanged and avoid over-correction, previous sequence-to-sequence (seq2seq) models generate results from scratch, which are not guaranteed to follow the original sentence structure and may suffer from the over-correction problem. In the meantime, the recently proposed sequence tagging models can overcome the over-correction problem by only generating edit operations, but are conditioned on human designed language-specific tagging labels. In this paper, we combine the pros and alleviate the cons of both models by proposing a novel Sequence-to-Action (S2A) module. The S2A module jointly takes the source and target sentences as input, and is able to automatically generate a token-level action sequence before predicting each token, where each action is generated from three choices named \texttt{SKIP}, \texttt{COPY} and \texttt{GEN}erate. Then the actions are fused with the basic seq2seq framework to provide final predictions. We conduct experiments on the benchmark datasets of both English and Chinese GEC tasks. Our model consistently outperforms the seq2seq baselines, while being able to significantly alleviate the over-correction problem as well as holding better generality and diversity in the generation results compared to the sequence tagging models.
  </details>

+ **Frustratingly Easy System Combination for Grammatical Error Correction**
  + Authors: Muhammad Qorib, Seung-Hoon Na, Hwee Tou Ng
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.143/
  + Code: https://github.com/nusnlp/esc
  + <details>
      <summary>Abstract</summary>
      In this paper, we formulate system combination for grammatical error correction (GEC) as a simple machine learning task: binary classification. We demonstrate that with the right problem formulation, a simple logistic regression algorithm can be highly effective for combining GEC models. Our method successfully increases the F0.5 score from the highest base GEC system by 4.2 points on the CoNLL-2014 test set and 7.2 points on the BEA-2019 test set. Furthermore, our method outperforms the state of the art by 4.0 points on the BEA-2019 test set, 1.2 points on the CoNLL-2014 test set with original annotation, and 3.4 points on the CoNLL-2014 test set with alternative annotation. We also show that our system combination generates better corrections with higher F0.5 scores than the conventional ensemble.
    </details>
[//]: # (+ Key Words: Ensembling; Edit Type; Linear Regression; Application Oriented)

+ **[`zh,en,ja`] Position Offset Label Prediction for Grammatical Error Correction**
  + Authors: Xiuyu Wu, Jingsong Yu, Xu Sun, Yunfang Wu
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.480/
  + Code: Not released yet.
  + <details>
      <summary>Abstract</summary>
      We introduce a novel position offset label prediction subtask to the encoder-decoder architecture for grammatical error correction (GEC) task. To keep the meaning of the input sentence unchanged, only a few words should be inserted or deleted during correction, and most of tokens in the erroneous sentence appear in the paired correct sentence with limited position movement. Inspired by this observation, we design an auxiliary task to predict position offset label (POL) of tokens, which is naturally capable of integrating different correction editing operations into a unified framework. Based on the predicted POL, we further propose a new copy mechanism (P-copy) to replace the vanilla copy module. Experimental results on Chinese, English and Japanese datasets demonstrate that our proposed POL-Pc framework obviously improves the performance of baseline models. Moreover, our model yields consistent performance gain over various data augmentation methods. Especially, after incorporating synthetic data, our model achieves a 38.95 F-0.5 score on Chinese GEC dataset, which outperforms the previous state-of-the-art by a wide margin of 1.98 points.
    </details>

+ **[`zh`] String Editing Based Chinese Grammatical Error Diagnosis**
  + Authors: Haihua Xie, Xiaoqing Lyu, Xuefei Chen
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.474/
  + Code: https://github.com/xiebimsa/se-cged
  + <details>
      <summary>Abstract</summary>
      Chinese Grammatical Error Diagnosis (CGED) suffers the problems of numerous types of grammatical errors and insufficiency of training data. In this paper, we propose a string editing based CGED model that requires less training data by using a unified workflow to handle various types of grammatical errors. Two measures are proposed in our model to enhance the performance of CGED. First, the detection and correction of grammatical errors are divided into different stages. In the stage of error detection, the model only outputs the types of grammatical errors so that the tag vocabulary size is significantly reduced compared with other string editing based models. Secondly, the correction of some grammatical errors is converted to the task of masked character inference, which has plenty of training data and mature solutions. Experiments on datasets of NLPTEA-CGED demonstrate that our model outperforms other CGED models in many aspects.
    </details>

+ <a name="sun-et-al-ijcal2022"></a>**[`en,zh,de,ru`] A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-Lingual Language Model**
  + Authors: Xin Sun, Tao Ge, Shuming Ma, Jingjing Li, Furu Wei, Houfeng Wang
  + Conference: IJCAI
  + Link: https://www.ijcai.org/proceedings/2022/0606
  + <details>
      <summary>Abstract</summary>
      Synthetic data construction of Grammatical Error Correction (GEC) for non-English languages relies heavily on human-designed and language-specific rules, which produce limited error-corrected patterns. In this paper, we propose a generic and language-independent strategy for multilingual GEC, which can train a GEC system effectively for a new non-English language with only two easy-to-access resources: 1) a pre-trained cross-lingual language model (PXLM) and 2) parallel translation data between English and the language. Our approach creates diverse parallel GEC data without any language-specific operations by taking the non-autoregressive translation generated by PXLM and the gold translation as error-corrected sentence pairs. Then, we reuse PXLM to initialize the GEC model and pre-train it with the synthetic data generated by itself, which yields further improvement. We evaluate our approach on three public benchmarks of GEC in different languages. It achieves the state-of-the-art results on the NLPCC 2018 Task 2 dataset (Chinese) and obtains competitive performance on Falko-Merlin (German) and RULEC-GEC (Russian). Further analysis demonstrates that our data construction method is complementary to rule-based approaches.
    </details>
[//]: # (+ Keywords: Language-agnostic Data Augmentation; Pre-trained Language Models)

+ **Dynamic Negative Example Construction for Grammatical Error Correction using Contrastive Learning**
  + Authors: He Junyi, Zhuang Junbin, Li Xia
  + Conference: CCL
  + Link: https://aclanthology.org/2022.ccl-1.83/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) aims at correcting texts with different types of grammatical errors into natural and correct forms. Due to the difference of error type distribution and error density, current grammatical error correction systems may over-correct writings and produce a low precision. To address this issue, in this paper, we propose a dynamic negative example construction method for grammatical error correction using contrastive learning. The proposed method can construct sufficient negative examples with diverse grammatical errors, and can be dynamically used during model training. The constructed negative examples are beneficial for the GEC model to correct sentences precisely and suppress the model from over-correction. Experimental results show that our proposed method enhances model precision, proving the effectiveness of our method.
    </details>
  
+ **[`el`] Enriching Grammatical Error Correction Resources for Modern Greek**
  + Authors: Katerina Korre, John Pavlopoulos
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.532/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC), a task of Natural Language Processing (NLP), is challenging for underepresented languages. This issue is most prominent in languages other than English. This paper addresses the issue of data and system sparsity for GEC purposes in the modern Greek Language. Following the most popular current approaches in GEC, we develop and test an MT5 multilingual text-to-text transformer for Greek. To our knowledge this the first attempt to create a fully-fledged GEC model for Greek. Our evaluation shows that our system reaches up to 52.63% F0.5 score on part of the Greek Native Corpus (GNC), which is 16% below the winning system of the BEA-19 shared task on English GEC. In addition, we provide an extended version of the Greek Learner Corpus (GLC), on which our model reaches up to 22.76% F0.5. Previous versions did not include corrections with the annotations which hindered the potential development of efficient GEC systems. For that reason we provide a new set of corrections. This new dataset facilitates an exploration of the generalisation abilities and robustness of our system, given that the assessment is conducted on learner data while the training on native data.
    </details>

+ <a name="li-et-al-ipm2022"></a>**Incorporating rich syntax information in Grammatical Error Correction**
  + Authors: Zuchao Li, Kevin Parnow, Hai Zhao
  + Journal: Information Processing & Management
  + Link: https://www.sciencedirect.com/science/article/pii/S0306457322000206
  + <details>
      <summary>Abstract</summary>
      Syntax parse trees are a method of representing sentence structure and are often used to provide models with syntax information and enhance downstream task performance. Because grammar and syntax are inherently linked, the incorporation of syntax parse trees in GEC is a natural solution. In this work, we present a method of incorporating syntax parse trees for Grammatical Error Correction (GEC). Building off a strong sequence-to-sequence Transformer baseline, we present a unified parse integration method for GEC that allows for the use of both dependency and constituency parse trees, as well as their combination - a syntactic graph. Specifically, on the sentence encoder, we propose a graph encoder that can encode dependency trees and constituent trees at the same time, yielding two representations for terminal nodes (i.e., the token of the sentence) and non-terminal nodes. We next use two cross-attentions (NT-Cross-Attention and T-Cross-Attention) to aggregate these source syntactic representations to the target side for final corrections prediction. In addition to evaluating our models on the popular CoNLL-2014 Shared Task and JFLEG GEC benchmarks, we affirm the effectiveness of our proposed method by testing both varying levels of parsing quality and exploring the use of both parsing formalisms. With further empirical exploration and analysis to identify the source of improvement, we found that rich syntax information provided clear clues for GEC; a syntactic graph composed of multiple syntactic parse trees can effectively compensate for the limited quality and insufficient error correction capability of a single syntactic parse tree.
    </details>
 
+ <a name="pajak-and-pajak-esa2022"></a>**[`ar,zh,cs,de,en,ro,ru`] Multilingual fine-tuning for Grammatical Error Correction**
  + Authors: Krzysztof Pająk, Dominik Pająk
  + Journal: Expert Systems with Applications
  + Link: https://www.sciencedirect.com/science/article/pii/S0957417422003773?ref=pdf_download&fr=RR-9&rr=7ad7dd3fcb910466
  + <details>
      <summary>Abstract</summary>
      Finding a single model capable of comprehending multiple languages is an area of active research in Natural Language Processing (NLP). Recently developed models such as mBART, mT5 or xProphetNet can solve problems connected with, for instance, machine translation and summarization for many languages. However, good multilingual solutions to the problem of Grammatical Error Correction (GEC) are still missing — this paper aims at filling this gap. We first review current annotated GEC datasets and then apply existing pre-trained multilingual models to correct grammatical errors in multiple languages. In our experiments, we compare how different pre-training approaches impact the final GEC quality. Our result is a single model that can correct seven different languages and is the best (in terms of F-score) currently reported multilingual GEC model. Additionally, our multilingual model achieves better results than the SOTA monolingual model for Romanian.
    </details>

+ <a name="zhang-et-al-2022"></a>**CSynGEC: Incorporating Constituent-based Syntax for Grammatical Error Correction with a Tailored GEC-Oriented Parser**
  + Authors: Yue Zhang, Zhenghua Li
  + Link: https://arxiv.org/abs/2211.08158
  + <details>
      <summary>Abstract</summary>
      Recently, Zhang et al. (2022) propose a syntax-aware grammatical error correction (GEC) approach, named SynGEC, showing that incorporating tailored dependency-based syntax of the input sentence is quite beneficial to GEC. This work considers another mainstream syntax formalism, i.e., constituent-based syntax. By drawing on the successful experience of SynGEC, we first propose an extended constituent-based syntax scheme to accommodate errors in ungrammatical sentences. Then, we automatically obtain constituency trees of ungrammatical sentences to train a GEC-oriented constituency parser by using parallel GEC data as a pivot. For syntax encoding, we employ the graph convolutional network (GCN). Experimental results show that our method, named CSynGEC, yields substantial improvements over strong baselines. Moreover, we investigate the integration of constituent-based and dependency-based syntax for GEC in two ways: 1) intra-model combination, which means using separate GCNs to encode both kinds of syntax for decoding in a single model; 2)inter-model combination, which means gathering and selecting edits predicted by different models to achieve final corrections. We find that the former method improves recall over using one standalone syntax formalism while the latter improves precision, and both lead to better F0.5 values.
    </details>

### GEC Datasets
+ **ErAConD: Error Annotated Conversational Dialog Dataset for Grammatical Error Correction**
  + Authors: Xun Yuan, Derek Pham, Sam Davidson, Zhou Yu
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.5/
  + Code: https://github.com/yuanxun-yx/eracond
  + <details>
      <summary>Abstract</summary>
      Currently available grammatical error correction (GEC) datasets are compiled using essays or other long-form text written by language learners, limiting the applicability of these datasets to other domains such as informal writing and conversational dialog. In this paper, we present a novel GEC dataset consisting of parallel original and corrected utterances drawn from open-domain chatbot conversations; this dataset is, to our knowledge, the first GEC dataset targeted to a human-machine conversational setting. We also present a detailed annotation scheme which ranks errors by perceived impact on comprehension, making our dataset more representative of real-world language learning applications. To demonstrate the utility of the dataset, we use our annotated data to fine-tune a state-of-the-art GEC model. Experimental results show the effectiveness of our data in improving GEC model performance in a conversational scenario.
    </details>

+ **FCGEC: Fine-Grained Corpus for Chinese Grammatical Error Correction**
  + Authors: Lvxiaowei Xu, Jianwang Wu, Jiawei Peng, Jiayu Fu, Ming Cai
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.137/
  + Code: https://github.com/xlxwalex/FCGEC
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) has been broadly applied in automatic correction and proofreading system recently. However, it is still immature in Chinese GEC due to limited high-quality data from native speakers in terms of category and scale. In this paper, we present FCGEC, a fine-grained corpus to detect, identify and correct the grammatical errors. FCGEC is a human-annotated corpus with multiple references, consisting of 41,340 sentences collected mainly from multi-choice questions in public school Chinese examinations. Furthermore, we propose a Switch-Tagger-Generator (STG) baseline model to correct the grammatical errors in low-resource settings. Compared to other GEC benchmark models, experimental results illustrate that STG outperforms them on our FCGEC. However, there exists a significant gap between benchmark models and humans that encourages future models to bridge it.
    </details>
  
+ **[`zh`] MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction**
  + Authors: Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, Min Zhang
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.227/
  + Code: https://github.com/HillZhang1999/MuCGEC
  + <details>
      <summary>Abstract</summary>
      This paper presents MuCGEC, a multi-reference multi-source evaluation dataset for Chinese Grammatical Error Correction (CGEC), consisting of 7,063 sentences collected from three Chinese-as-a-Second-Language (CSL) learner sources. Each sentence is corrected by three annotators, and their corrections are carefully reviewed by a senior annotator, resulting in 2.3 references per sentence. We conduct experiments with two mainstream CGEC models, i.e., the sequence-to-sequence model and the sequence-to-edit model, both enhanced with large pretrained language models, achieving competitive benchmark performance on previous and our datasets. We also discuss CGEC evaluation methodologies, including the effect of multiple references and using a char-based metric. Our annotation guidelines, data, and code are available at https://github.com/HillZhang1999/MuCGEC.
    </details>

+ **[`cs`] Czech Grammar Error Correction with a Large and Diverse Corpus**
  + Authors: Jakub Náplava, Milan Straka, Jana Straková, Alexandr Rosen
  + Conference: TACL
  + Link: https://aclanthology.org/2022.tacl-1.26/
  + Code: https://github.com/ufal/errant_czech
  + <details>
      <summary>Abstract</summary>
      We introduce a large and diverse Czech corpus annotated for grammatical error correction (GEC) with the aim to contribute to the still scarce data resources in this domain for languages other than English. The Grammar Error Correction Corpus for Czech (GECCC) offers a variety of four domains, covering error distributions ranging from high error density essays written by non-native speakers, to website texts, where errors are expected to be much less common. We compare several Czech GEC systems, including several Transformer-based ones, setting a strong baseline to future research. Finally, we meta-evaluate common GEC metrics against human judgments on our data. We make the new Czech GEC corpus publicly available under the CC BY-SA 4.0 license at http://hdl.handle.net/11234/1-4639.
    </details>

+ **[`ja`] Construction of a Quality Estimation Dataset for Automatic Evaluation of Japanese Grammatical Error Correction**
  + Authors: Daisuke Suzuki, Yujin Takahashi, Ikumi Yamashita, Taichi Aida, Tosho Hirasawa, Michitaka Nakatsuji, Masato Mita, Mamoru Komachi
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.596/
  + <details>
      <summary>Abstract</summary>
      In grammatical error correction (GEC), automatic evaluation is considered as an important factor for research and development of GEC systems. Previous studies on automatic evaluation have shown that quality estimation models built from datasets with manual evaluation can achieve high performance in automatic evaluation of English GEC. However, quality estimation models have not yet been studied in Japanese, because there are no datasets for constructing quality estimation models. In this study, therefore, we created a quality estimation dataset with manual evaluation to build an automatic evaluation model for Japanese GEC. By building a quality estimation model using this dataset and conducting a meta-evaluation, we verified the usefulness of the quality estimation model for Japanese GEC.
    </details>
  
+ **ProQE: Proficiency-wise Quality Estimation dataset for Grammatical Error Correction**
  + Authors: Yujin Takahashi, Masahiro Kaneko, Masato Mita, Mamoru Komachi
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.644/
  + <details>
      <summary>Abstract</summary>
      This study investigates how supervised quality estimation (QE) models of grammatical error correction (GEC) are affected by the learners’ proficiency with the data. QE models for GEC evaluations in prior work have obtained a high correlation with manual evaluations. However, when functioning in a real-world context, the data used for the reported results have limitations because prior works were biased toward data by learners with relatively high proficiency levels. To address this issue, we created a QE dataset that includes multiple proficiency levels and explored the necessity of performing proficiency-wise evaluation for QE of GEC. Our experiments demonstrated that differences in evaluation dataset proficiency affect the performance of QE models, and proficiency-wise evaluation helps create more robust models.
    </details>
  
+ **Improving Grammatical Error Correction for Multiword Expressions**
  + Authors: Shiva Taslimipoor, Christopher Bryant, Zheng Yuan
  + Conference: MWE
  + Link: https://aclanthology.org/2022.lrec-1.644/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is the task of automatically correcting errors in text. It has mainly been developed to assist language learning, but can also be applied to native text. This paper reports on preliminary work in improving GEC for multiword expression (MWE) error correction. We propose two systems which incorporate MWE information in two different ways: one is a multi-encoder decoder system which encodes MWE tags in a second encoder, and the other is a BART pre-trained transformer-based system that encodes MWE representations using special tokens. We show improvements in correcting specific types of verbal MWEs based on a modified version of a standard GEC evaluation approach.
    </details>
  
### GEC Evaluation
+ **Revisiting Grammatical Error Correction Evaluation and Beyond**
  + Authors: Peiyuan Gong, Xuebo Liu, Heyan Huang, Min Zhang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.463/
  + Code: https://github.com/pygongnlp/PT-M2
  + <details>
    <summary>Abstract</summary>
    Pretraining-based (PT-based) automatic evaluation metrics (e.g., BERTScore and BARTScore) have been widely used in several sentence generation tasks (e.g., machine translation and text summarization) due to their better correlation with human judgments over traditional overlap-based methods. Although PT-based methods have become the de facto standard for training grammatical error correction (GEC) systems, GEC evaluation still does not benefit from pretrained knowledge. This paper takes the first step towards understanding and improving GEC evaluation with pretraining. We first find that arbitrarily applying PT-based metrics to GEC evaluation brings unsatisfactory correlation results because of the excessive attention to inessential systems outputs (e.g., unchanged parts). To alleviate the limitation, we propose a novel GEC evaluation metric to achieve the best of both worlds, namely PT-M2 which only uses PT-based metrics to score those corrected parts. Experimental results on the CoNLL14 evaluation task show that PT-M2 significantly outperforms existing methods, achieving a new state-of-the-art result of 0.949 Pearson correlation. Further analysis reveals that PT-M2 is robust to evaluate competitive GEC systems. Source code and scripts are freely available at https://github.com/pygongnlp/PT-M2.
  </details>

+ **Grammatical Error Correction: Are We There Yet?**
  + Authors: Muhammad Reza Qorib, Hwee Tou Ng
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.246/
  + <details>
      <summary>Abstract</summary>
      There has been much recent progress in natural language processing, and grammatical error correction (GEC) is no exception. We found that state-of-the-art GEC systems (T5 and GECToR) outperform humans by a wide margin on the CoNLL-2014 test set, a benchmark GEC test corpus, as measured by the standard F0.5 evaluation metric. However, a careful examination of their outputs reveals that there are still classes of errors that they fail to correct. This suggests that creating new test data that more accurately measure the true performance of GEC systems constitutes important future work.
    </details>

+ **Grammatical Error Correction Systems for Automated Assessment: Are They Susceptible to Universal Adversarial Attacks?**
  + Authors: Vyas Raina, Yiting Lu, Mark Gales
  + Conference: AACL
  + Link: https://aclanthology.org/2022.aacl-main.13/
  + Code: https://github.com/rainavyas/gec-universal-attack
  + <details>
    <summary>Abstract</summary>
    Grammatical error correction (GEC) systems are a useful tool for assessing a learner’s writing ability. These systems allow the grammatical proficiency of a candidate’s text to be assessed without requiring an examiner or teacher to read the text. A simple summary of a candidate’s ability can be measured by the total number of edits between the input text and the GEC system output: the fewer the edits the better the candidate. With advances in deep learning, GEC systems have become increasingly powerful and accurate. However, deep learning systems are susceptible to adversarial attacks, in which a small change at the input can cause large, undesired changes at the output. In the context of GEC for automated assessment, the aim of an attack can be to deceive the system into not correcting (concealing) grammatical errors to create the perception of higher language ability. An interesting aspect of adversarial attacks in this scenario is that the attack needs to be simple as it must be applied by, for example, a learner of English. The form of realistic attack examined in this work is appending the same phrase to each input sentence: a concatenative universal attack. The candidate only needs to learn a single attack phrase. State-of-the-art GEC systems are found to be susceptible to this form of simple attack, which transfers to different test sets as well as system architectures.
  </details>

### GED
+ **[`zh`] Improving Chinese Grammatical Error Detection via Data augmentation by Conditional Error Generation**
  + Authors: Tianchi Yue, Shulin Liu, Huihui Cai, Tao Yang, Shengkang Song, TingHao Yu
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2022.findings-acl.233/
  + Code: https://github.com/tc-yue/DA_CGED
  + <details>
      <summary>Abstract</summary>
      Chinese Grammatical Error Detection(CGED) aims at detecting grammatical errors in Chinese texts. One of the main challenges for CGED is the lack of annotated data. To alleviate this problem, previous studies proposed various methods to automatically generate more training samples, which can be roughly categorized into rule-based methods and model-based methods. The rule-based methods construct erroneous sentences by directly introducing noises into original sentences. However, the introduced noises are usually context-independent, which are quite different from those made by humans. The model-based methods utilize generative models to imitate human errors. The generative model may bring too many changes to the original sentences and generate semantically ambiguous sentences, so it is difficult to detect grammatical errors in these generated sentences. In addition, generated sentences may be error-free and thus become noisy data. To handle these problems, we propose CNEG, a novel Conditional Non-Autoregressive Error Generation model for generating Chinese grammatical errors. Specifically, in order to generate a context-dependent error, we first mask a span in a correct text, then predict an erroneous span conditioned on both the masked text and the correct span. Furthermore, we filter out error-free spans by measuring their perplexities in the original sentences. Experimental results show that our proposed method achieves better performance than all compared data augmentation methods on the CGED-2018 and CGED-2020 benchmarks.
    </details>
[//]: # (+ Key Words: Generative CGED; BERT Masking; Conditional Error Generation )

+ **Exploring the Capacity of a Large-scale Masked Language Model to Recognize Grammatical Errors**
  + Authors: Ryo Nagata, Manabu Kimura, Kazuaki Hanawa
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2022.findings-acl.324/
  + Code: https://github.com/tc-yue/DA_CGED
  + <details>
      <summary>Abstract</summary>
      Abstract
      In this paper, we explore the capacity of a language model-based method for grammatical error detection in detail. We first show that 5 to 10% of training data are enough for a BERT-based error detection method to achieve performance equivalent to what a non-language model-based method can achieve with the full training data; recall improves much faster with respect to training data size in the BERT-based method than in the non-language model method. This suggests that (i) the BERT-based method should have a good knowledge of the grammar required to recognize certain types of error and that (ii) it can transform the knowledge into error detection rules by fine-tuning with few training samples, which explains its high generalization ability in grammatical error detection. We further show with pseudo error data that it actually exhibits such nice properties in learning rules for recognizing various types of error. Finally, based on these findings, we discuss a cost-effective method for detecting grammatical errors with feedback comments explaining relevant grammatical rules to learners.
    </details>

### SGED
+ **On Assessing and Developing Spoken ’Grammatical Error Correction’ Systems**
  + Authors: Yiting Lu, Stefano Bannò, Mark Gales
  + Conference: BEA
  + Link: https://aclanthology.org/2022.bea-1.9/
  + <details>
      <summary>Abstract</summary>
      Spoken ‘grammatical error correction’ (SGEC) is an important process to provide feedback for second language learning. Due to a lack of end-to-end training data, SGEC is often implemented as a cascaded, modular system, consisting of speech recognition, disfluency removal, and grammatical error correction (GEC). This cascaded structure enables efficient use of training data for each module. It is, however, difficult to compare and evaluate the performance of individual modules as preceeding modules may introduce errors. For example the GEC module input depends on the output of non-native speech recognition and disfluency detection, both challenging tasks for learner data.This paper focuses on the assessment and development of SGEC systems. We first discuss metrics for evaluating SGEC, both individual modules and the overall system. The system-level metrics enable tuning for optimal system performance. A known issue in cascaded systems is error propagation between modules.To mitigate this problem semi-supervised approaches and self-distillation are investigated. Lastly, when SGEC system gets deployed it is important to give accurate feedback to users. Thus, we apply filtering to remove edits with low-confidence, aiming to improve overall feedback precision. The performance metrics are examined on a Linguaskill multi-level data set, which includes the original non-native speech, manual transcriptions and reference grammatical error corrections, to enable system analysis and development.
    </details>

## Papers in 2021
### GEC Methods
+ <a name="rothe-et-al-acl2021"></a>**[`en,de,ru,cs`] A Simple Recipe for Multilingual Grammatical Error Correction**
  + Authors: Sascha Rothe, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn
  + Conference: ACL
  + Link: https://aclanthology.org/2021.acl-short.89/
  + Code: https://github.com/google-research-datasets/clang8 (only the clang8 data are provided)
  + <details>
      <summary>Abstract</summary>
      This paper presents a simple recipe to trainstate-of-the-art multilingual Grammatical Error Correction (GEC) models. We achieve this by first proposing a language-agnostic method to generate a large number of synthetic examples. The second ingredient is to use large-scale multilingual language models (up to 11B parameters). Once fine-tuned on language-specific supervised sets we surpass the previous state-of-the-art results on GEC benchmarks in four languages: English, Czech, German and Russian. Having established a new set of baselines for GEC, we make our results easily reproducible and accessible by releasing a CLANG-8 dataset. It is produced by using our best model, which we call gT5, to clean the targets of a widely used yet noisy Lang-8 dataset. cLang-8 greatly simplifies typical GEC training pipelines composed of multiple fine-tuning stages – we demonstrate that performing a single fine-tuning stepon cLang-8 with the off-the-shelf language models yields further accuracy improvements over an already top-performing gT5 model for English.
    </details>
[//]: # (+ Keywords: Language-agnostic Data Augmentation; Pre-trained Language Models; Distillation)

+ <a name="flachs-et-al-bea2021"></a>**[`es,de,ru,cs`]Data Strategies for Low-Resource Grammatical Error Correction**
  + Authors: Simon Flachs, Felix Stahlberg, Shankar Kumar
  + Conference: BEA
  + Link: https://aclanthology.org/2021.bea-1.12/
  + <details>
      <summary>Abstract</summary>
   Grammatical Error Correction (GEC) is a task that has been extensively investigated for the English language. However, for low-resource languages the best practices for training GEC systems have not yet been systematically determined. We investigate how best to take advantage of existing data sources for improving GEC systems for languages with limited quantities of high quality training data. We show that methods for generating artificial training data for GEC can benefit from including morphological errors. We also demonstrate that noisy error correction data gathered from Wikipedia revision histories and the language learning website Lang8, are valuable data sources. Finally, we show that GEC systems pre-trained on noisy data sources can be fine-tuned effectively using small amounts of high quality, human-annotated data.
   </details>

+ <a name="straka-et-al-wnut2021"></a>**[`cs,de,ru`] Character Transformations for Non-Autoregressive GEC Tagging**
  + Authors: Milan Straka, Jakub Náplava, Jana Straková
  + Conference: EMNLP-WNUT
  + Link: https://aclanthology.org/2021.wnut-1.46/
  + Code: https://github.com/ufal/wnut2021_character_transformations_gec
  + <details>
      <summary>Abstract</summary>
   We propose a character-based non-autoregressive GEC approach, with automatically generated character transformations. Recently, per-word classification of correction edits has proven an efficient, parallelizable alternative to current encoder-decoder GEC systems. We show that word replacement edits may be suboptimal and lead to explosion of rules for spelling, diacritization and errors in morphologically rich languages, and propose a method for generating character transformations from GEC corpus. Finally, we train character transformation models for Czech, German and Russian, reaching solid results and dramatic speedup compared to autoregressive systems. The source code is released at https://github.com/ufal/wnut2021_character_transformations_gec.
   </details>

+ <a name="wan-and-wan-2021"></a>**A Syntax-Guided Grammatical Error Correction Model with Dependency Tree Correction**
  + Authors: Zhaohong Wan, Xiaojun Wan
  + Link: https://arxiv.org/abs/2111.03294
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is a task of detecting and correcting grammatical errors in sentences. Recently, neural machine translation systems have become popular approaches for this task. However, these methods lack the use of syntactic knowledge which plays an important role in the correction of grammatical errors. In this work, we propose a syntax-guided GEC model (SG-GEC) which adopts the graph attention mechanism to utilize the syntactic knowledge of dependency trees. Considering the dependency trees of the grammatically incorrect source sentences might provide incorrect syntactic knowledge, we propose a dependency tree correction task to deal with it. Combining with data augmentation method, our model achieves strong performances without using any large pre-trained models. We evaluate our model on public benchmarks of GEC task and it achieves competitive results.
    </details>

## Papers in 2020
### GEC Methods
+ <a name="yamashita-et-al-coling2020"></a>**[`en,ru,cs`] Cross-lingual Transfer Learning for Grammatical Error Correction**
  + Authors: Ikumi Yamashita, Satoru Katsumata, Masahiro Kaneko, Aizhan Imankulova, Mamoru Komachi
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.415/
  + <details>
      <summary>Abstract</summary>
      In this study, we explore cross-lingual transfer learning in grammatical error correction (GEC) tasks. Many languages lack the resources required to train GEC models. Cross-lingual transfer learning from high-resource languages (the source models) is effective for training models of low-resource languages (the target models) for various tasks. However, in GEC tasks, the possibility of transferring grammatical knowledge (e.g., grammatical functions) across languages is not evident. Therefore, we investigate cross-lingual transfer learning methods for GEC. Our results demonstrate that transfer learning from other languages can improve the accuracy of GEC. We also demonstrate that proximity to source languages has a significant impact on the accuracy of correcting certain types of errors.
    </details>

+ <a name="katsusama-and-komachi-aacl2020"></a>**[`en,de,cs,ru`] Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model**
  + Authors: Satoru Katsumata, Mamoru Komachi
  + Conference: AACL
  + Link: https://aclanthology.org/2020.aacl-main.83/
  + Code: https://github.com/Katsumata420/generic-pretrained-GEC
  + <details>
      <summary>Abstract</summary>
      Studies on grammatical error correction (GEC) have reported on the effectiveness of pretraining a Seq2Seq model with a large amount of pseudodata. However, this approach requires time-consuming pretraining of GEC because of the size of the pseudodata. In this study, we explored the utility of bidirectional and auto-regressive transformers (BART) as a generic pretrained encoder-decoder model for GEC. With the use of this generic pretrained model for GEC, the time-consuming pretraining can be eliminated. We find that monolingual and multilingual BART models achieve high performance in GEC, with one of the results being comparable to the current strong results in English GEC.
    </details>

## Papers in 2019
### GEC Methods
+ <a name="naplava-and-straka-wnut2019"></a>**[`en,cs,de,ru`] Grammatical Error Correction in Low-Resource Scenarios**
  + Authors: Jakub Náplava, Milan Straka
  + Conference: EMNLP-WNUT
  + Link: https://aclanthology.org/D19-5545/
  + Code: https://github.com/ufal/low-resource-gec-wnut2019
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction in English is a long studied problem with many existing systems and datasets. However, there has been only a limited research on error correction of other languages. In this paper, we present a new dataset AKCES-GEC on grammatical error correction for Czech. We then make experiments on Czech, German and Russian and show that when utilizing synthetic parallel corpus, Transformer neural machine translation model can reach new state-of-the-art results on these datasets. AKCES-GEC is published under CC BY-NC-SA 4.0 license at http://hdl.handle.net/11234/1-3057, and the source code of the GEC model is available at https://github.com/ufal/low-resource-gec-wnut2019.
    </details>

## Papers in 2018
### GEC Methods
+ **[`de`] Using Wikipedia Edits in Low Resource Grammatical Error Correction**
  + Authors: Adriane Boyd
  + Conference: EMNLP-WNUT
  + Link: https://aclanthology.org/W18-6111/
  + Code: https://github.com/adrianeboyd/ boyd-wnut2018/
  + <details>
      <summary>Abstract</summary>
      We develop a grammatical error correction (GEC) system for German using a small gold GEC corpus augmented with edits extracted from Wikipedia revision history. We extend the automatic error annotation tool ERRANT (Bryant et al., 2017) for German and use it to analyze both gold GEC corrections and Wikipedia edits (Grundkiewicz and Junczys-Dowmunt, 2014) in order to select as additional training data Wikipedia edits containing grammatical corrections similar to those in the gold corpus. Using a multilayer convolutional encoder-decoder neural network GEC approach (Chollampatt and Ng, 2018), we evaluate the contribution of Wikipedia edits and find that carefully selected Wikipedia edits increase performance by over 5%.
    </details>

## Topics
### Multi-lingual
+ [A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-Lingual Language Model](#sun-et-al-ijcal2022)
  + Year: 2022
  + Languages: `en,zh,de,ru`
  + <details>
    <summary>Method: language-agnostic data augmentation, multi-lingual pre-trained language models.</summary>
    
    + A two-stage training strategy is employed: (1) pre-training with augmented pseudo data and (2) fine-tuning with language-specific annotated data.
  
    + In the pre-training stage, this paper uses an NAT (non-autoregressive translation) model to translate sentences, and uses the translated sentences as erroneous sentences. The erroneous sentences are paired with the corresponding correct sentences as augmentation pairs. Specifically, let the target language for data augmentation be `zh` (Chinese), the augmentation method is as follows. In this manner, 10M augmented pairs in every language are constructed to pre-train the GEC model. Additionally, the GEC model is initialized with the pre-trained weights of DeltaLM (an InfoXLM-initialized encoder-decoder model) before pre-training.
      1) A translation pair (`en`, `zh`) is obtained from machine translation data.
      2) Word-level noises (insertion, deletion, replacement, swapping) are injected to the `zh` sentence, obtaining `zh^1`. 
      3) Some words in the `zh^1` sentence are masked, obtaining `zh^2`.
      Then, InfoXLM (an NAT) receives the (`en`, `zh^2`) pair as input and outputs the translated sentence `zh^2^` with distributional sampling.
      4) The outputted sentence `zh^2^` is further noised with character-level noises (insertion, deletion, replacement, swapping, casing), obtaining `zh^3`. Finally, (`zh^3`, `zh`) is treated as an augmented pair for pre-training.
  
    + In the fine-tuning stage, the pre-trained model is fine-tuned with language-specific annotated data. 
  </details>

+ [Multilingual fine-tuning for Grammatical Error Correction](#pajak-and-pajak-esa2022)
  + Year: 2022
  + Languages: `ar,zh,cs,de,en,ro,ru `

+ [A Simple Recipe for Multilingual Grammatical Error Correction](#rothe-et-al-acl2021)
  + Year: 2021
  + Languages: `en,de,ru,cs`
  + <details>
    <summary>Method: language-agnostic data augmentation, multi-lingual pre-trained language models.</summary>
    
    + A two-stage training strategy is employed: (1) pre-training with augmented pseudo data and (2) fine-tuning with language-specific annotated data.
  
    + In the pre-training stage, augmented data are constructed from the mC4 corpus that contains unlabeled texts in 101 languages. Each noised sentence is paired with the original sentence to form an augmented pair (sent_noised, sent_original). The augmented pairs are used to pre-train the mT5 model.
    Specifically, the unlabeled texts are noised with the following operations:
      1) dropping spans of tokens, swapping tokens
      2) dropping spans of characters, swapping characters, inserting characters, lower-casing a word, upper-casing the first character of a word

    + In the fine-tuning stage, the pre-trained model is further fine-tuned with language-specific data.
  
  </details>
 
+ [Data Strategies for Low-Resource Grammatical Error Correction](#flachs-et-al-bea2021)
  + Year: 2021
  + Languages: `es,de,ru,cs`

+ [ Character Transformations for Non-Autoregressive GEC Tagging](#straka-et-al-wnut2021)
  + Year: 2021
  + Languages: `cs,de,ru`

+ [Cross-lingual Transfer Learning for Grammatical Error Correction](#yamashita-et-al-coling2020)
  + Year: 2020
  + Languages: `en,ru,cs`

+ [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model](#katsusama-and-komachi-aacl2020)
  + Year: 2020
  + Languages: `en,de,cs,ru`

+ [Grammatical Error Correction in Low-Resource Scenarios](#naplava-and-straka-wnut2019)
  + Year: 2019
  + Languages: `en,cs,de,ru`

### Syntax-Enhanced

+ [SynGEC: Syntax-Enhanced Grammatical Error Correction with a Tailored GEC-Oriented Parser](#zhang-et-al-emnlp2022)
  + Year: 2022
  + Method: GEC-oriented dependency syntax parser + GCN fusing. 

+ [CSynGEC: Incorporating Constituent-based Syntax for Grammatical Error Correction with a Tailored GEC-Oriented Parser](#zhang-et-al-2022)
  + Year: 2022
  + Method: GEC-oriented constituency syntax parser + GCN fusing; incorporation of dependency & constituency syntax.

+ [Incorporating rich syntax information in Grammatical Error Correction](#li-et-al-ipm2022)
  + Year: 2022

+ [A Syntax-Guided Grammatical Error Correction Model with Dependency Tree Correction](#wan-and-wan-2021)
  + Year: 2021
  + Method: Dependency syntax + GAT fusing; dependency tree correction (prediction of dependency relation, distance and ancestor-descendant relation).

[//]: # (---)

[//]: # (**The papers below will be re-arranged.**)

[//]: # (## GEC)

[//]: # (<!-- - [x] 2021/1/6 [Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction]&#40;#bert-gec&#41; [ACL-2020] √)

[//]: # (- [x] 2021/1/6 [GECToR - Grammatical Error Correction: Tag, Not Rewrite]&#40;#gector&#41; [ACL-2020] √)

[//]: # (- [x] 2021/1/7 [MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking]&#40;#maskgec&#41; [AAAI-2020])

[//]: # (- [x] 2021/1/7 [Towards Minimal Supervision BERT-Based Grammar Error Correction &#40;Student Abstract&#41;]&#40;#minimal-supervision&#41; [AAAI-2020])

[//]: # (- [x] 2021/1/7 [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model]&#40;#bart-gec&#41; [AACL-2020] √)

[//]: # (- [x] 2021/1/9 [Chinese Grammatical Correction Using BERT-based Pre-trained Model]&#40;#chinese-bert-gec&#41; [IJCNLP-2020])

[//]: # (- [x] 2021/1/10 [Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction]&#40;#efficiency&#41; [EMNLP-2020])

[//]: # (- [x] 2021/1/10 [Heterogeneous Recycle Generation for Chinese Grammatical Correction]&#40;#heterogeneous&#41; [COLING-2020] √)

[//]: # (- [x] 2021/1/10 [TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020]&#40;#chinese-bert-init&#41; [AACL-2020])

[//]: # (- [x] 2021/1/11 [Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction]&#40;#local-beam-search&#41; [COLING-2020])

[//]: # (- [x] 2021/1/12 [Seq2Edits: Sequence Transduction Using Span-level Edit Operations]&#40;#seq2edits&#41; [EMNLP-2020])

[//]: # (- [x] 2021/1/12 [Adversarial Grammatical Error Correction]&#40;#adversarial&#41; [EMNLP-2020])

[//]: # (- [x] 2021/1/17 Pseudo-Bidirectional Decoding for Local Sequence Transduction [EMNLP-2020])

[//]: # (- [x] 2021/1/18 Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data [ACL-2019])

[//]: # (- [x] 2021/1/18 An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction [ACL-2019])

[//]: # (- [x] 2021/1/19 Parallel Iterative Edit Models for Local Sequence Transduction [EMNLP-2019])

[//]: # (- [x] 2021/1/19 Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data [NAACL-2019])

[//]: # (- [x] 2021/1/20 A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning [ACL-2020])

[//]: # (- [x] 2021/1/20 The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction [ACL-2019])

[//]: # (- [x] 2021/1/20 TMU Transformer System Using BERT for Re-ranking at BEA 2019 Grammatical Error Correction on Restricted Track [ACL-2019])

[//]: # (- [x] 2021/1/21 Noisy Channel for Low Resource Grammatical Error Correction [ACL-2019])

[//]: # (- [x] 2021/1/22 The BLCU System in the BEA 2019 Shared Task [ACL-2019])

[//]: # (- [x] 2021/1/22 The AIP-Tohoku System at the BEA-2019 Shared Task [ACL-2019])

[//]: # (- [x] 2021/1/22 CUNI System for the Building Educational Applications 2019 Shared Task: Grammatical Error Correction [ACL-2019] -->)

[//]: # ()
[//]: # (| Index | Date | Paper | Conference | Code | Note |)

[//]: # (| :-: | --- | --- | --- | --- | --- |)

[//]: # (| 1* | 21/1/6 | Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction &#40;Kaneko et al.&#41; | ACL-2020 | [Code]&#40;https://github.com/kanekomasahiro/bert-gec&#41; | [Note]&#40;#bert-gec&#41; |)

[//]: # (| 2* | 21/1/6 | GECToR - Grammatical Error Correction: Tag, Not Rewrite &#40;Omelianchuk et al.&#41; | ACL-2020 | [Code]&#40;https://github.com/grammarly/gector&#41; | [Note]&#40;#gector&#41; |)

[//]: # (| 3* | 21/1/7 | MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking &#40;Zhao and Wang&#41; | AAAI-2020 |  | [Note]&#40;#maskgec&#41; |)

[//]: # (| 4 | 21/1/7 | Towards Minimal Supervision BERT-Based Grammar Error Correction &#40;Student Abstract&#41; &#40;Li et al.&#41; | AAAI-2020 |  | [Note]&#40;#minimal-supervision&#41; |)

[//]: # (| 5* | 21/1/7 | Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model &#40;Katsumata and Komachi&#41; | AACL-2020 | [Code]&#40;https://github.com/Katsumata420/generic-pretrained-GEC&#41; | [Note]&#40;#bart-gec&#41; |)

[//]: # (| 6 | 21/1/9 | Chinese Grammatical Correction Using BERT-based Pre-trained Model &#40;Wang et al.&#41; | IJCNLP-2020 |  | [Note]&#40;#chinese-bert-gec&#41; |)

[//]: # (| 7* | 21/1/10 | Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction &#40;Chen et al.&#41; | EMNLP-2020 |  | [Note]&#40;#efficiency&#41; |)

[//]: # (| 8* | 21/1/10 | Heterogeneous Recycle Generation for Chinese Grammatical Correction &#40;Hinson et al.&#41; | COLING-2020 |  | [Note]&#40;#heterogeneous&#41; |)

[//]: # (| 9 | 21/1/10 | TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020 &#40;Wang and Komachi&#41; | AACL-2020 |  | [Note]&#40;#chinese-bert-init&#41; |)

[//]: # (| 10 | 21/1/11 | Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction &#40;Hotate et al.&#41; | COLING-2020 |  | [Note]&#40;#local-beam-search&#41; |)

[//]: # (| 11 | 21/1/12 | Seq2Edits: Sequence Transduction Using Span-level Edit Operations &#40;Stahlberg and Kumar&#41; | EMNLP-2020 | [Code]&#40;https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/transformer_seq2edits.py&#41; | [Note]&#40;#seq2edits&#41; |)

[//]: # (| 12 | 21/1/12 | Adversarial Grammatical Error Correction &#40;Raheja and Alikaniotis&#41; | EMNLP-2020 |  | [Note]&#40;#adversarial&#41; |)

[//]: # (| 13* | 21/1/17 | Pseudo-Bidirectional Decoding for Local Sequence Transduction &#40;Zhou et al.&#41; | EMNLP-2020 |  |  |)

[//]: # (| 14 | 21/1/18 | Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data &#40;Grundkiewicz et al.&#41; | ACL-2019 |  |  |)

[//]: # (| 15 | 21/1/18 | An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction &#40;Kiyono et al.&#41; | ACL-2019 |  |  |)

[//]: # (| 16 | 21/1/19 | Parallel Iterative Edit Models for Local Sequence Transduction &#40;Awasthi et al.&#41; | EMNLP-2019 |  |  |)

[//]: # (| 17 | 21/1/19 | Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data &#40;Zhao et al.&#41; | NAACL-2019 |  |  |)

[//]: # (| 18 | 21/1/20 | A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning &#40;Choe et al.&#41; | ACL-2020 |  |  |)

[//]: # (| 19 | 21/1/20 | The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction &#40;Alikaniotis and Raheja&#41; | ACL-2019 |  |  |)

[//]: # (| 20 | 21/1/20 | TMU Transformer System Using BERT for Re-ranking at BEA 2019 Grammatical Error Correction on Restricted Track &#40;Kaneko et al.&#41; | ACL-2019 |  |  |)

[//]: # (| 21 | 21/1/21 | Noisy Channel for Low Resource Grammatical Error Correction &#40;Flachs et al.&#41; | ACL-2019 |  |  |)

[//]: # (| 22 | 21/1/22 | The BLCU System in the BEA 2019 Shared Task &#40;Yang et al.&#41; | ACL-2019 |  |  |)

[//]: # (| 23 | 21/1/22 | The AIP-Tohoku System at the BEA-2019 Shared Task &#40;Asano et al.&#41; | ACL-2019 |  |  |)

[//]: # (| 24 | 21/1/22 | CUNI System for the Building Educational Applications 2019 Shared Task: Grammatical Error Correction &#40;Náplava and Straka&#41; | ACL-2019 |  |  |)

[//]: # (| 25 | 21/1/27 | Cross-Sentence Grammatical Error Correction &#40;Chollampatt et al.&#41; | ACL-2019 |  |  |)

[//]: # ()
[//]: # (## GED)

[//]: # (<!-- - [x] 2021/1/6 [基于数据增强和多任务特征学习的中文语法错误检测方法]&#40;#chinese-multi-task&#41; [CCL-2020] √)

[//]: # (- [x] 2021/1/11 [Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis]&#40;#score-based&#41; [AACL-2020])

[//]: # (- [x] 2021/1/11 [CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared]&#40;#bert-crf&#41; [AACL-2020])

[//]: # (- [x] 2021/1/11 [Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis]&#40;#resnet-bert&#41; [AACL-2020])

[//]: # (- [x] 2021/1/11 [Chinese Grammatical Errors Diagnosis System Based on BERT at NLPTEA-2020 CGED Shared Task]&#40;#bert-bilstm-crf-3gram-seq2seq&#41; [AACL-2020])

[//]: # (- [x] 2021/1/11 [Chinese Grammatical Error Detection Based on BERT Model]&#40;#bert-finetuned&#41; [AACL-2020])

[//]: # (- [x] 2021/1/21 Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection [CICLING-2019] -->)

[//]: # ()
[//]: # (| Index | Date | Paper | Conference | Code | Note |)

[//]: # (| :-: | --- | --- | --- | --- | --- |)

[//]: # (| 1* | 21/1/6 | 基于数据增强和多任务特征学习的中文语法错误检测方法 &#40;Xie et al.&#41; | CCL-2020 |  | [Note]&#40;#chinese-multi-task&#41; |)

[//]: # (| 2 | 21/1/11 | Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis &#40;Cao et al.&#41; | AACL-2020 |  | [Note]&#40;#score-based&#41; |)

[//]: # (| 3 | 21/1/11 | CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared &#40;Wu and Wang&#41; | AACL-2020 |  | [Note]&#40;#bert-crf&#41; |)

[//]: # (| 4 | 21/1/11 | Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis &#40;Wang et al.&#41; | AACL-2020 |  | [Note]&#40;#resnet-bert&#41; |)

[//]: # (| 5 | 21/1/11 | Chinese Grammatical Errors Diagnosis System Based on BERT at NLPTEA-2020 CGED Shared Task &#40;Zan et al.&#41; | AACL-2020 |  | [Note]&#40;#bert-bilstm-crf-3gram-seq2seq&#41; |)

[//]: # (| 6 | 21/1/11 | Chinese Grammatical Error Detection Based on BERT Model &#40;Cheng and Duan&#41; | AACL-2020 |  | [Note]&#40;#bert-finetuned&#41; |)

[//]: # (| 7 | 21/1/21 | Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection &#40;Kaneko et al.&#41; | CICLING-2019 |  |  |)

[//]: # ()
[//]: # (## DA)

[//]: # (<!-- - [x] Improving Grammatical Error Correction with Machine Translation Pairs [EMNLP-2020])

[//]: # (- [x] A Self-Refinement Strategy for Noise Reduction in Grammatical Error Correction [EMNLP-2020] -->)

[//]: # ()
[//]: # (| Index | Date | Paper | Conference | Code | Note |)

[//]: # (| :-: | --- | --- | --- | --- | --- |)

[//]: # (| 1 | 21/1/11 | A Self-Refinement Strategy for Noise Reduction in Grammatical Error Correction &#40;Mita et al.&#41; | EMNLP-2020 |  |  |)

[//]: # ()
[//]: # (## Related)

[//]: # (<!-- - [x] 2021/1/5 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)

[//]: # (- [x] 2021/1/5 [Incorporating BERT into Neural Machine Translation]&#40;#bert-nmt&#41; [ICLR-2020] √)

[//]: # (- [x] 2021/1/17 Agreement on Target-Bidirectional LSTMs for Sequence-to-Sequence Learning [AAAI-2016])

[//]: # (- [x] 2021/1/17 Agreement on Target-bidirectional Neural Machine Translation [NAACL-2016])

[//]: # (- [x] 2021/1/17 Edinburgh Neural Machine Translation Systems for WMT 16)

[//]: # (- [x] 2021/1/22 LIMIT-BERT: Linguistic Informed Multi-Task BERT [EMNLP-2020])

[//]: # (- [x] 2021/1/23 Distilling Knowledge Learned in BERT for Text Generation [ACL-2020])

[//]: # (- [x] 2021/1/23 Towards Making the Most of BERT in Neural Machine Translation [AAAI-2020])

[//]: # (- [x] 2021/1/23 Acquiring Knowledge from Pre-Trained Model to Neural Machine Translation [AAAI-2020] -->)

[//]: # ()
[//]: # (| Index | Date | Paper | Conference | Code | Note |)

[//]: # (| :-: | --- | --- | --- | --- | --- |)

[//]: # (| 1 | 21/1/5 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding &#40;Devlin et al.&#41; | NAACL-2019 |  |  |)

[//]: # (| 2* | 21/1/5 | Incorporating BERT into Neural Machine Translation &#40;Zhu et al.&#41; | ICLR-2020 | [Code]&#40;https://github.com/bert-nmt/bert-nmt&#41; | [Note]&#40;#bert-nmt&#41; |)

[//]: # (| 3 | 21/1/17 | Agreement on Target-Bidirectional LSTMs for Sequence-to-Sequence Learning &#40;Liu et al.&#41; | AAAI-2016 |  |  |)

[//]: # (| 4 | 21/1/17 | Agreement on Target-bidirectional Neural Machine Translation &#40;Liu et al.&#41; | NAACL-2016 |  |  |)

[//]: # (| 5* | 21/1/17 | Edinburgh Neural Machine Translation Systems for WMT 16 &#40;Sennrich et al.&#41; | WMT-2016 |  |  |)

[//]: # (| 6 | 21/1/22 | LIMIT-BERT: Linguistic Informed Multi-Task BERT &#40;Zhou et al.&#41; | EMNLP-2020 |  |  |)

[//]: # (| 7 | 21/1/23 | Distilling Knowledge Learned in BERT for Text Generation &#40;Chen et al.&#41; | ACL-2020 |  |  |)

[//]: # (| 8 | 21/1/23 | Towards Making the Most of BERT in Neural Machine Translation &#40;Yang et al.&#41; | AAAI-2020 |  |  |)

[//]: # (| 9 | 21/1/23 | Acquiring Knowledge from Pre-Trained Model to Neural Machine Translation &#40;Weng et al.&#41; | AAAI-2020 |  |  |)

[//]: # (| 10 | 21/1/26 | Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting &#40;Zhou et al.&#41; | - |  |  |)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## Seq2Seq)

[//]: # (1. <span id="bert-gec">[ACL-2020] Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction</span>  )

[//]: # (Applied the BERT-fused model for GEC. The BERT is finetuned with MLM and GED to fix the inconsistent input distribution between the raw data for BERT training and the GEC data. Pseudo-data and R2L are also used for performance boosting.  )

[//]: # (https://github.com/kanekomasahiro/bert-gec)

[//]: # ()
[//]: # (2. <span id="chinese-bert-gec">[IJCNLP-2020] Chinese Grammatical Correction Using BERT-based Pre-trained Model  )

[//]: # (Tries BERT-init &#40;BERT-encoder in the papar&#41; and BERT-fused for Chinese GEC. The Chinese GEC ver. of *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, even less techniques used.)

[//]: # ()
[//]: # (3. <span id="bart-gec">[AACL-2020] Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model</span>  )

[//]: # (Used BART for GEC and says that BART can be a baseline for GEC, which can reach high performance by simple finetuning with GEC data instead of pseudo-data pretraining.  )

[//]: # (https://github.com/Katsumata420/generic-pretrained-GEC)

[//]: # ()
[//]: # (4. <span id="efficiency">[EMNLP-2020] Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction</span>  )

[//]: # (Combines a sequence tagging model for erroneous span detection and a seq2seq model for erroneous span correction to make the GEC process more efficient. The sequence tagging model &#40;BERT-like&#41; looks for spans needing to be corrected by outputting binary vectors, and the seq2seq model receives inputs annotated according to the outputs of the sequence tagging model and only produces outputs corresponding to the detected spans. Pseudo-data is used for pre-training the ESD and ESC models.)

[//]: # ()
[//]: # (## Seq2Edits)

[//]: # (1. <span id="seq2edits">[EMNLP-2020] Seq2Edits: Sequence Transduction Using Span-level Edit Operations</span>  )

[//]: # (Proposes a method for tasks containing many overlaps such as GEC. Uses Transformer with the decoder modified. The model receives a source sentence and at each inference time-step outputs a 3-tuple  which corresponds to an edit operation &#40;error tag, source span end position, replacement&#41;. The error tag provides clear explanation. The paper conducts experiments on 5 NLP tasks containing many overlaps. Experiments with and without pretraining are conducted.  )

[//]: # (&#40;Not very clear about the modified decoder.&#41;  )

[//]: # (https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/transformer_seq2edits.py)

[//]: # ()
[//]: # (## Seq Labeling)

[//]: # (1. <span id="gector">[ACL-2020] GECToR - Grammatical Error Correction: Tag, Not Rewrite</span>  )

[//]: # (Used a BERT sequence tagger. Developed custom task-specific g-transformations such as CASE, MERGE and so on. Since each time a token in the source sentence can only map an edit, iterative correction may be required. A 3-stage training strategy is used: data-aug pretraining - finetuning on err data - finetuning on err and err-free data.  )

[//]: # (https://github.com/grammarly/gector)

[//]: # ()
[//]: # (2. <span id="minimal-supervision">[AAAI-2020] Towards Minimal Supervision BERT-Based Grammar Error Correction &#40;Student Abstract&#41;</span>  )

[//]: # (Divides the GEC task into two stages: error identification and error correction. The first stage is a sequence tagging &#40;remain, substitution, ...&#41; task and a BERT is used for the second stage &#40;correction&#41;.   )

[//]: # (&#40;Not very clear about the method proposed by the paper.&#41;)

[//]: # ()
[//]: # (## Pipeline)

[//]: # (1. <span id="heterogeneous">[COLING-2020] Heterogeneous Recycle Generation for Chinese Grammatical Correction</span>  )

[//]: # (Makes use of a sequence editing model, a seq2seq model and a spell checker to correct different kinds of errors &#40;small scale errors, large scale errors and spell errors respectively&#41;. Iterative decoding is applied on &#40;sequence editing model, seq2seq model&#41;. The proposed method needs not data-aug but still achieves comparable performance.)

[//]: # ()
[//]: # (## Multi-Task Learning)

[//]: # (1. <span id="chinese-multi-task">[GED] [CCL-2020] 基于数据增强和多任务特征学习的中文语法错误检测方法</span>  )

[//]: # (Implements Chinese GED through data-augmentation and pretrained BERT finetuned using multi-task learning. The data-augmentation method applied here is simple, including manipulations such as insertions, deletions and so on. Some rules are designed to maintain the meanings of sentences. The Chinese BERT is used for GED with a CRF layer on top. It is finetuned through multi-task learning: pos tagging, parsing and grammar error detection.)

[//]: # ()
[//]: # (## Beam Search)

[//]: # (1. <span id="local-beam-search">[COLING-2020] Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction</span>  )

[//]: # (Proposes a local beam search method to output diverse outputs. The proposed method generates more diverse outputs than the plain beam search, and only modifies where should be corrected rather than changing the whole sequence as the global beam search. The copy factor in the copy-augmented Transformer is used as a penalty score.)

[//]: # ()
[//]: # (## Adversarial Training)

[//]: # (1. <span id="adversarial">[EMNLP-2020] Adversarial Grammatical Error Correction</span>  )

[//]: # (The first approach to use adversarial training for GEC. Uses a seq2seq model as the generator and a sentence-pair classification model for the discriminator. The discriminator basically acts as a novel evaluation method for evaluating the outputs generated by the generator, which directly models the task. No other technique such as data augmentation is used.  )

[//]: # (&#40;Not very clear about the adversarial training.&#41;)

[//]: # ()
[//]: # (## Dynamic Masking)

[//]: # (1. <span id="maskgec">[AAAI-2020] MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking</span>  )

[//]: # (Proposed a dynamic masking method for data-augmentation and generalization boosting. In each epoch each sentence is introduced noises with a prob by some manipulations, including padding substitution, random substution, word frequency substitution and so on.)

[//]: # ()
[//]: # (## NLPTEA)

[//]: # (1. <span id="chinese-bert-init">[AACL-2020] TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020  )

[//]: # (Uses BERT-init as in *Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction*, which is also the same as the BERT-encoder in *Chinese Grammatical Correction Using BERT-based Pre-trained Model*.)

[//]: # ()
[//]: # (2. <span id="score-based">[GED] [AACL-2020] Integrating BERT and Score-based Feature Gates for Chinese Grammatical Error Diagnosis</span>  )

[//]: # (Uses BiLSTM-CRF for GED, whose input is features concat composed of output of BERT, POS, POS score and PMI score. The scores are incorporated using a gating mechanism to avoid losing partial-order relationships when embedding continuous feature items.  )

[//]: # (&#40;Not very clear about the features used and the purpose of the gating mechanism.&#41;)

[//]: # ()
[//]: # (3. <span id="bert-crf">[GED] [AACL-2020] CYUT Team Chinese Grammatical Error Diagnosis System Report in NLPTEA-2020 CGED Shared]</span>  )

[//]: # (Uses BERT + CRF.)

[//]: # ()
[//]: # (4. <span id="resnet-bert">[GED] [AACL-2020] Combining ResNet and Transformer for Chinese Grammatical Error Diagnosis</span>  )

[//]: # (Applies res on BERT for GED. The encoded hidden repr is added with the emd and fed into the output layer.  )

[//]: # (&#40;Also related to GEC but not detailed, thus catogorize as GED.&#41;)

[//]: # ()
[//]: # (5. <span id="bert-bilstm-crf-3gram-seq2seq">[GED] [AACL-2020] Chinese Grammatical Errors Diagnosis System Based on BERT at NLPTEA-2020 CGED Shared Task</span>  )

[//]: # (Uses BERT-BiLSTM-CRF for GED. Uses a hybrid system containing a 3-gram and a seq2seq for GEC.)

[//]: # ()
[//]: # (6. <span id="bert-finetuned">[GED] [AACL-2020] Chinese Grammatical Error Detection Based on BERT Model</span>  )

[//]: # (Uses BERT finetuned on GEC datasets.)

[//]: # ()
[//]: # (## Related)

[//]: # (1. <span id="bert-nmt">[NMT] [ICLR-2020] Incorporating BERT into Neural Machine Translation</span>  )

[//]: # (Proposed a BERT-fused model. Comparing with the Vanilla Transformer, the proposed model has additionally one BERT-Enc Attention module in the encoder and a BERT-Dec Attention module in the decoder. Both of the additional modules are for incorporating features extracted by BERT whose weights are fixed. A Vanilla Transformer is trained in the first training stage, and in the second stage the BERT and additional modules are trained together.  )

[//]: # (https://github.com/bert-nmt/bert-nmt)
