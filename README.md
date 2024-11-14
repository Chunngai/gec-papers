# GEC Papers
Paper list for grammatical error correction (GEC) from 2018 to now.

## 2024
+ **Classist Tools: Social Class Correlates with Performance in NLP**
  + Authors: Amanda Curry, Giuseppe Attanasio, Zeerak Talat, Dirk Hovy
  + Conference: ACL
  + Link: https://aclanthology.org/2024.acl-long.682/
  + <details>
      <summary>Abstract</summary>
      The field of sociolinguistics has studied factors affecting language use for the last century. Labov (1964) and Bernstein (1960) showed that socioeconomic class strongly influences our accents, syntax and lexicon. However, despite growing concerns surrounding fairness and bias in Natural Language Processing (NLP), there is a dearth of studies delving into the effects it may have on NLP systems. We show empirically that NLP systems’ performance is affected by speakers’ SES, potentially disadvantaging less-privileged socioeconomic groups. We annotate a corpus of 95K utterances from movies with social class, ethnicity and geographical language variety and measure the performance of NLP systems on three tasks: language modelling, automatic speech recognition, and grammar error correction. We find significant performance disparities that can be attributed to socioeconomic status as well as ethnicity and geographical differences. With NLP technologies becoming ever more ubiquitous and quotidian, they must accommodate all language varieties to avoid disadvantaging already marginalised groups. We argue for the inclusion of socioeconomic class in future language technologies.
    </details>
+ **Detection-Correction Structure via General Language Model for Grammatical Error Correction**
  + Authors: Wei Li, Houfeng Wang
  + Conference: ACL
  + Link: https://aclanthology.org/2024.acl-long.96/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is a task dedicated to rectifying texts with minimal edits, which can be decoupled into two components: detection and correction. However, previous works have predominantly focused on direct correction, with no prior efforts to integrate both into a single model. Moreover, the exploration of the detection-correction paradigm by large language models (LLMs) remains underdeveloped. This paper introduces an integrated detection-correction structure, named DeCoGLM, based on the General Language Model (GLM). The detection phase employs a fault-tolerant detection template, while the correction phase leverages autoregressive mask infilling for localized error correction. Through the strategic organization of input tokens and modification of attention masks, we facilitate multi-task learning within a single model. Our model demonstrates competitive performance against the state-of-the-art models on English and Chinese GEC datasets. Further experiments present the effectiveness of the detection-correction structure in LLMs, suggesting a promising direction for GEC.
    </details>
+ **In-context Mixing (ICM): Code-mixed Prompts for Multilingual LLMs**
  + Authors: Bhavani Shankar, Preethi Jyothi, Pushpak Bhattacharyya
  + Conference: ACL
  + Link: https://aclanthology.org/2024.acl-long.228/
  + <details>
      <summary>Abstract</summary>
      We introduce a simple and effective prompting technique called in-context mixing (ICM) for effective in-context learning (ICL) with multilingual large language models (MLLMs). With ICM, we modify the few-shot examples within ICL prompts to be intra-sententially code-mixed by randomly swapping content words in the target languages with their English translations. We observe that ICM prompts yield superior performance in NLP tasks such as disfluency correction, grammar error correction and text simplification that demand a close correspondence between the input and output sequences. Significant improvements are observed mainly for low-resource languages that are under-represented during the pretraining and finetuning of MLLMs. We present an extensive set of experiments to analyze when ICM is effective and what design choices contribute towards its effectiveness. ICM works consistently and significantly better than other prompting techniques across models of varying capacity such as mT0-XXL, BloomZ and GPT-4.
    </details>
+ **Alirector: Alignment-Enhanced Chinese Grammatical Error Corrector**
  + Authors: Haihui Yang, Xiaojun Quan
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2024.findings-acl.148/
  + <details>
      <summary>Abstract</summary>
      Chinese grammatical error correction (CGEC) faces serious overcorrection challenges when employing autoregressive generative models such as sequence-to-sequence (Seq2Seq) models and decoder-only large language models (LLMs). While previous methods aim to address overcorrection in Seq2Seq models, they are difficult to adapt to decoder-only LLMs. In this paper, we propose an alignment-enhanced corrector for the overcorrection problem that applies to both Seq2Seq models and decoder-only LLMs. Our method first trains a correction model to generate an initial correction of the source sentence. Then, we combine the source sentence with the initial correction and feed it through an alignment model for another round of correction, aiming to enforce the alignment model to focus on potential overcorrection. Moreover, to enhance the model’s ability to identify nuances, we further explore the reverse alignment of the source sentence and the initial correction. Finally, we transfer the alignment knowledge from two alignment models to the correction model, instructing it on how to avoid overcorrection. Experimental results on three CGEC datasets demonstrate the effectiveness of our approach in alleviating overcorrection and improving overall performance. Our code has been made publicly available.
    </details>
+ **Improving Grammatical Error Correction via Contextual Data Augmentation**
  + Authors: Yixuan Wang, Baoxin Wang, Yijun Liu, Qingfu Zhu, Dayong Wu, Wanxiang Che
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2024.findings-acl.647/
  + <details>
      <summary>Abstract</summary>
      Nowadays, data augmentation through synthetic data has been widely used in the field of Grammatical Error Correction (GEC) to alleviate the problem of data scarcity. However, these synthetic data are mainly used in the pre-training phase rather than the data-limited fine tuning phase due to inconsistent error distribution and noisy labels. In this paper, we propose a synthetic data construction method based on contextual augmentation, which can ensure an efficient augmentation of the original data with a more consistent error distribution. Specifically, we combine rule-based substitution with model-based generation, using the generation model to generate a richer context for the extracted error patterns. Besides, we also propose a relabeling-based data cleaning method to mitigate the effects of noisy labels in synthetic data. Experiments on CoNLL14 and BEA19-Test show that our proposed augmentation method consistently and substantially outperforms strong baselines and achieves the state-of-the-art level with only a few synthetic data.
    </details>
+ **Likelihood-based Mitigation of Evaluation Bias in Large Language Models**
  + Authors: Masanari Ohi, Masahiro Kaneko, Ryuto Koike, Mengsay Loem, Naoaki Okazaki
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2024.findings-acl.193/
  + <details>
      <summary>Abstract</summary>
      Large Language Models (LLMs) are widely used to evaluate natural language generation tasks as automated metrics.However, the likelihood, a measure of LLM’s plausibility for a sentence, can vary due to superficial differences in sentences, such as word order and sentence structure.It is therefore possible that there might be a likelihood bias if LLMs are used for evaluation: they might overrate sentences with higher likelihoods while underrating those with lower likelihoods.In this paper, we investigate the presence and impact of likelihood bias in LLM-based evaluators.We also propose a method to mitigate the likelihood bias.Our method utilizes highly biased instances as few-shot examples for in-context learning.Our experiments in evaluating the data-to-text and grammatical error correction tasks reveal that several LLMs we test display a likelihood bias.Furthermore, our proposed method successfully mitigates this bias, also improving evaluation performance (in terms of correlation of models with human scores) significantly.
    </details>
+ **Prompting open-source and commercial language models for grammatical error correction of English learner text**
  + Authors: Christopher Davis, Andrew Caines, O Andersen, Shiva Taslimipoor, Helen Yannakoudakis, Zheng Yuan, Christopher Bryant, Marek Rei, Paula Buttery
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2024.findings-acl.711/
  + <details>
      <summary>Abstract</summary>
      Thanks to recent advances in generative AI, we are able to prompt large language models (LLMs) to produce texts which are fluent and grammatical. In addition, it has been shown that we can elicit attempts at grammatical error correction (GEC) from LLMs when prompted with ungrammatical input sentences. We evaluate how well LLMs can perform at GEC by measuring their performance on established benchmark datasets. We go beyond previous studies, which only examined GPT* models on a selection of English GEC datasets, by evaluating seven open-source and three commercial LLMs on four established GEC benchmarks. We investigate model performance and report results against individual error types. Our results indicate that LLMs do not always outperform supervised English GEC models except in specific contexts – namely commercial LLMs on benchmarks annotated with fluency corrections as opposed to minimal edits. We find that several open-source models outperform commercial ones on minimal edit benchmarks, and that in some settings zero-shot prompting is just as competitive as few-shot prompting.
    </details>
+ **Towards Better Utilization of Multi-Reference Training Data for Chinese Grammatical Error Correction**
  + Authors: Yumeng Liu, Zhenghua Li, HaoChen Jiang, Bo Zhang, Chen Li, Ji Zhang
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2024.findings-acl.180/
  + <details>
      <summary>Abstract</summary>
      For the grammatical error correction (GEC) task, there usually exist multiple correction ways for an erroneous input sentence, leading to multiple references. Observing the high proportion of multi-reference instances in Chinese GEC training data, we target a systematic study on how to better utilize multi-reference training data. We propose two new approaches and a simple two-stage training strategy. We compare them against previously proposed approaches, on two Chinese training datasets, i.e., Lang-8 for second language learner texts and FCGEC-Train for native speaker texts, and three test datasets. The experiments and analyses demonstrate the effectiveness of our proposed approaches and reveal interesting insights. Our code is available at https://github.com/ymliucs/MrGEC.
    </details>
+ **LLM-based Code-Switched Text Generation for Grammatical Error Correction**
  + Authors: Tom Potter, Zheng Yuan
  + Conference: EMNLP
  + Link: https://aclanthology.org/2024.emnlp-main.942/
  + <details>
      <summary>Abstract</summary>
      With the rise of globalisation, code-switching (CSW) has become a ubiquitous part of multilingual conversation, posing new challenges for natural language processing (NLP), especially in Grammatical Error Correction (GEC). This work explores the complexities of applying GEC systems to CSW texts. Our objectives include evaluating the performance of state-of-the-art GEC systems on an authentic CSW dataset from English as a Second Language (ESL) learners, exploring synthetic data generation as a solution to data scarcity, and developing a model capable of correcting grammatical errors in monolingual and CSW texts. We generated synthetic CSW GEC data, resulting in one of the first substantial datasets for this task, and showed that a model trained on this data is capable of significant improvements over existing systems. This work targets ESL learners, aiming to provide educational technologies that aid in the development of their English grammatical correctness without constraining their natural multilingualism.
    </details>
+ **Multi-pass Decoding for Grammatical Error Correction**
  + Authors: Xiaoying Wang, Lingling Mu, Jingyi Zhang, Hongfei Xu
  + Conference: EMNLP
  + Link: https://aclanthology.org/2024.emnlp-main.553/
  + <details>
      <summary>Abstract</summary>
      Sequence-to-sequence (seq2seq) models achieve comparable or better grammatical error correction performance compared to sequence-to-edit (seq2edit) models. Seq2edit models normally iteratively refine the correction result, while seq2seq models decode only once without aware of subsequent tokens. Iteratively refining the correction results of seq2seq models via Multi-Pass Decoding (MPD) may lead to better performance. However, MPD increases the inference costs. Deleting or replacing corrections in previous rounds may lose useful information in the source input. We present an early-stop mechanism to alleviate the efficiency issue. To address the source information loss issue, we propose to merge the source input with the previous round correction result into one sequence. Experiments on the CoNLL-14 test set and BEA-19 test set show that our approach can lead to consistent and significant improvements over strong BART and T5 baselines (+1.80, +1.35, and +2.02 F0.5 for BART 12-2, large and T5 large respectively on CoNLL-14 and +2.99, +1.82, and +2.79 correspondingly on BEA-19), obtaining F0.5 scores of 68.41 and 75.36 on CoNLL-14 and BEA-19 respectively.
    </details>
+ **Efficient and Interpretable Grammatical Error Correction with Mixture of Experts**
  + Authors: Muhammad Reza Qorib, Alham Fikri Aji, Hwee Tou Ng
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2024.findings-emnlp.997/
  + <details>
      <summary>Abstract</summary>
      Error type information has been widely used to improve the performance of grammatical error correction (GEC) models, whether for generating corrections, re-ranking them, or combining GEC models. Combining GEC models that have complementary strengths in correcting different error types is very effective in producing better corrections. However, system combination incurs a high computational cost due to the need to run inference on the base systems before running the combination method itself. Therefore, it would be more efficient to have a single model with multiple sub-networks that specialize in correcting different error types. In this paper, we propose a mixture-of-experts model, MoECE, for grammatical error correction. Our model successfully achieves the performance of T5-XL with three times fewer effective parameters. Additionally, our model produces interpretable corrections by also identifying the error type during inference.
    </details>
+ **Search if you don’t know! Knowledge-Augmented Korean Grammatical Error Correction with Large Language Models**
  + Authors: Seonmin Koo, Jinsung Kim, Chanjun Park, Heuiseok Lim
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2024.findings-emnlp.6/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) system is a practical task used in the real world, showing high achievements alongside the development of large language models (LLMs). However, these achievements have been primarily obtained in English, and there is a relative lack of performance for non-English data, such as Korean. We hypothesize that this insufficiency occurs because relying solely on the parametric knowledge of LLMs makes it difficult to thoroughly understand the given context in the Korean GEC. Therefore, we propose a Knowledge-Augmented GEC (KAGEC) framework that incorporates evidential information from external sources into the prompt for the GEC task. KAGEC first extracts salient phrases from the given source and retrieves non-parametric knowledge based on these phrases, aiming to enhance the context-aware generation capabilities of LLMs. Furthermore, we conduct validations for fine-grained error types to identify those requiring a retrieval-augmented manner when LLMs perform Korean GEC. According to experimental results, most LLMs, including ChatGPT, demonstrate significant performance improvements when applying KAGEC.
    </details>
+ **To Err Is Human, but Llamas Can Learn It Too**
  + Authors: Agnes Luhtaru, Taido Purason, Martin Vainikko, Maksym Del, Mark Fishel
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2024.findings-emnlp.727/
  + <details>
      <summary>Abstract</summary>
      This study explores enhancing grammatical error correction (GEC) through automatic error generation (AEG) using language models (LMs). Specifically, we fine-tune Llama 2 LMs for error generation and find that this approach yields synthetic errors akin to human errors. Next, we train GEC Llama models using these artificial errors and outperform previous state-of-the-art error correction models, with gains ranging between 0.8 and 6 F0.5 points across all tested languages (German, Ukrainian, and Estonian). Moreover, we demonstrate that generating errors by fine-tuning smaller sequence-to-sequence models and prompting large commercial LMs (GPT3.5 and GPT4) also results in synthetic errors beneficially affecting error generation models. We openly release trained models for error generation and correction as well as all the synthesized error datasets for the covered languages.
    </details>
+ **Towards Explainable Chinese Native Learner Essay Fluency Assessment: Dataset, Tasks, and Method**
  + Authors: Xinshu Shen, Hongyi Wu, Yadong Zhang, Man Lan, Xiaopeng Bai, Shaoguang Mao, Yuanbin Wu, Xinlin Zhuang, Li Cai
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2024.findings-emnlp.910/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is a crucial technique in Automated Essay Scoring (AES) for evaluating the fluency of essays. However, in Chinese, existing GEC datasets often fail to consider the importance of specific grammatical error types within compositional scenarios, lack research on data collected from native Chinese speakers, and largely overlook cross-sentence grammatical errors. Furthermore, the measurement of the overall fluency of an essay is often overlooked. To address these issues, we present CEFA (Chinese Essay Fluency Assessment), an extensive corpus that is derived from essays authored by native Chinese-speaking primary and secondary students and encapsulates essay fluency scores along with both coarse and fine-grained grammatical error types and corrections. Experiments employing various benchmark models on CEFA substantiate the challenge of our dataset. Our findings further highlight the significance of fine-grained annotations in fluency assessment and the mutually beneficial relationship between error types and corrections
    </details>
+ **Large Language Models Are State-of-the-Art Evaluator for Grammatical Error Correction**
  + Authors: Masamune Kobayashi, Masato Mita, Mamoru Komachi
  + Conference: NAACL
  + Link: https://aclanthology.org/2024.bea-1.6/
  + <details>
      <summary>Abstract</summary>
      Large Language Models (LLMs) have been reported to outperform existing automatic evaluation metrics in some tasks, such as text summarization and machine translation. However, there has been a lack of research on LLMs as evaluators in grammatical error correction (GEC). In this study, we investigate the performance of LLMs in GEC evaluation by employing prompts designed to incorporate various evaluation criteria inspired by previous research. Our extensive experimental results demonstrate that GPT-4 achieved Kendall’s rank correlation of 0.662 with human judgments, surpassing all existing methods. Furthermore, in recent GEC evaluations, we have underscored the significance of the LLMs scale and particularly emphasized the importance of fluency among evaluation criteria.
    </details>
+ **Pillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of Large Language Models**
  + Authors: Kostiantyn Omelianchuk, Andrii Liubonko, Oleksandr Skurzhanskyi, Artem Chernodub, Oleksandr Korniienko, Igor Samokhin
  + Conference: NAACL
  + Link: https://aclanthology.org/2024.bea-1.3/
  + <details>
      <summary>Abstract</summary>
      In this paper, we carry out experimental research on Grammatical Error Correction, delving into the nuances of single-model systems, comparing the efficiency of ensembling and ranking methods, and exploring the application of large language models to GEC as single-model systems, as parts of ensembles, and as ranking methods. We set new state-of-the-art records with F_0.5 scores of 72.8 on CoNLL-2014-test and 81.4 on BEA-test, respectively. To support further advancements in GEC and ensure the reproducibility of our research, we make our code, trained models, and systems’ outputs publicly available, facilitating future findings.
    </details>
+ **Synthetic Data Generation for Low-resource Grammatical Error Correction with Tagged Corruption Models**
  + Authors: Felix Stahlberg, Shankar Kumar
  + Conference: NAACL
  + Link: https://aclanthology.org/2024.bea-1.2/
  + <details>
      <summary>Abstract</summary>
      Tagged corruption models provide precise control over the introduction of grammatical errors into clean text. This capability has made them a powerful tool for generating pre-training data for grammatical error correction (GEC) in English. In this work, we demonstrate their application to four languages with substantially fewer GEC resources than English: German, Romanian, Russian, and Spanish. We release a new tagged-corruption dataset consisting of 2.5M examples per language that was generated by a fine-tuned PaLM 2 foundation model. Pre-training on tagged corruptions yields consistent gains across all four languages, especially for small model sizes and languages with limited human-labelled data.
    </details>
+ **Towards Automated Document Revision: Grammatical Error Correction, Fluency Edits, and Beyond**
  + Authors: Masato Mita, Keisuke Sakaguchi, Masato Hagiwara, Tomoya Mizumoto, Jun Suzuki, Kentaro Inui
  + Conference: NAACL
  + Link: https://aclanthology.org/2024.bea-1.21/
  + <details>
      <summary>Abstract</summary>
      Natural language processing (NLP) technology has rapidly improved automated grammatical error correction (GEC) tasks, and the GEC community has begun to explore document-level revision. However, there are two major obstacles to going beyond automated sentence-level GEC to NLP-based document-level revision support: (1) there are few public corpora with document-level revisions annotated by professional editors, and (2) it is infeasible to obtain all possible references and evaluate revision quality using such references because there are infinite revision possibilities. To address these challenges, this paper proposes a new document revision corpus, Text Revision of ACL papers (TETRA), in which multiple professional editors have revised academic papers sampled from the ACL anthology. This corpus enables us to focus on document-level and paragraph-level edits, such as edits related to coherence and consistency. Additionally, as a case study using the TETRA corpus, we investigate reference-less and interpretable methods for meta-evaluation to detect quality improvements according to document revisions. We show the uniqueness of TETRA compared with existing document revision corpora and demonstrate that a fine-tuned pre-trained language model can discriminate the quality of documents after revision even when the difference is subtle.
    </details>
+ **Ungrammatical-syntax-based In-context Example Selection for Grammatical Error Correction**
  + Authors: Chenming Tang, Fanyi Qu, Yunfang Wu
  + Conference: NAACL
  + Link: https://aclanthology.org/2024.naacl-long.99/
  + <details>
      <summary>Abstract</summary>
      In the era of large language models (LLMs), in-context learning (ICL) stands out as an effective prompting strategy that explores LLMs’ potency across various tasks. However, applying LLMs to grammatical error correction (GEC) is still a challenging task. In this paper, we propose a novel ungrammatical-syntax-based in-context example selection strategy for GEC. Specifically, we measure similarity of sentences based on their syntactic structures with diverse algorithms, and identify optimal ICL examples sharing the most similar ill-formed syntax to the test input. Additionally, we carry out a two-stage process to further improve the quality of selection results. On benchmark English GEC datasets, empirical results show that our proposed ungrammatical-syntax-based strategies outperform commonly-used word-matching or semantics-based methods with multiple LLMs. This indicates that for a syntax-oriented task like GEC, paying more attention to syntactic information can effectively boost LLMs’ performance. Our code is available at https://github.com/JamyDon/SynICL4GEC.
    </details>
+ **mEdIT: Multilingual Text Editing via Instruction Tuning**
  + Authors: Vipul Raheja, Dimitris Alikaniotis, Vivek Kulkarni, Bashar Alhafni, Dhruv Kumar
  + Conference: NAACL
  + Link: https://aclanthology.org/2024.naacl-long.56/
  + <details>
      <summary>Abstract</summary>
      We introduce mEdIT, a multi-lingual extension to CoEdIT – the recent state-of-the-art text editing models for writing assistance. mEdIT models are trained by fine-tuning multi-lingual large, pre-trained language models (LLMs) via instruction tuning. They are designed to take instructions from the user specifying the attributes of the desired text in the form of natural language instructions, such as “Grammatik korrigieren” (German) or “이 텍스 트를 단순화” (Korean). We build mEdIT by curating data from multiple publicly available human-annotated text editing datasets for three text editing tasks (Grammatical Error Correction (GEC), Text Simplification, and Paraphrasing) across diverse languages belonging to six different language families. We detail the design and training of mEdIT models and demonstrate their strong performance on many multi-lingual text editing benchmarks against other multilingual LLMs. We also find that mEdIT generalizes effectively to new languages over multilingual baselines. We publicly release our data, code, and trained models.
    </details>
+ **GEE! Grammar Error Explanation with Large Language Models**
  + Authors: Yixiao Song, Kalpesh Krishna, Rajesh Bhatt, Kevin Gimpel, Mohit Iyyer
  + Conference: NAACL Findings
  + Link: https://aclanthology.org/2024.findings-naacl.49/
  + <details>
      <summary>Abstract</summary>
      Existing grammatical error correction tools do not provide natural language explanations of the errors that they correct in user-written text. However, such explanations are essential for helping users learn the language by gaining a deeper understanding of its grammatical rules (DeKeyser, 2003; Ellis et al., 2006).To address this gap, we propose the task of grammar error explanation, where a system needs to provide one-sentence explanations for each grammatical error in a pair of erroneous and corrected sentences. The task is not easily solved by prompting LLMs: we find that, using one-shot prompting, GPT-4 only explains 40.6% of the errors and does not even attempt to explain 39.8% of the errors.Since LLMs struggle to identify grammar errors, we develop a two-step pipeline that leverages fine-tuned and prompted large language models to perform structured atomic token edit extraction, followed by prompting GPT-4 to explain each edit. We evaluate our pipeline on German, Chinese, and English grammar error correction data. Our atomic edit extraction achieves an F1 of 0.93 on German, 0.91 on Chinese, and 0.891 on English. Human evaluation of generated explanations reveals that 93.9% of German errors, 96.4% of Chinese errors, and 92.20% of English errors are correctly detected and explained. To encourage further research, we open-source our data and code.
    </details>
+ **Assessing the Efficacy of Grammar Error Correction: A Human Evaluation Approach in the Japanese Context**
  + Authors: Qiao Wang, Zheng Yuan
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.146/
  + <details>
      <summary>Abstract</summary>
      In this study, we evaluated the performance of the state-of-the-art sequence tagging grammar error detection and correction model (SeqTagger) using Japanese university students’ writing samples. With an automatic annotation toolkit, ERRANT, we first evaluated SeqTagger’s performance on error correction with human expert correction as the benchmark. Then a human-annotated approach was adopted to evaluate Seqtagger’s performance in error detection using a subset of the writing dataset. Results indicated a precision of 63.66% and a recall of 20.19% for error correction in the full dataset. For the subset, after manual exclusion of irrelevant errors such as semantic and mechanical ones, the model shows an adjusted precision of 97.98% and an adjusted recall of 42.98% for error detection, indicating the model’s high accuracy but also its conservativeness. Thematic analysis on errors undetected by the model revealed that determiners and articles, especially the latter, were predominant. Specifically, in terms of context-independent errors, the model occasionally overlooked basic ones and faced challenges with overly erroneous or complex structures. Meanwhile, context-dependent errors, notably those related to tense and noun number, as well as those possibly influenced by the students’ first language (L1), remained particularly challenging.
    </details>
+ **Controlled Generation with Prompt Insertion for Natural Language Explanations in Grammatical Error Correction**
  + Authors: Masahiro Kaneko, Naoaki Okazaki
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.350/
  + <details>
      <summary>Abstract</summary>
      In Grammatical Error Correction (GEC), it is crucial to ensure the user’s comprehension of a reason for correction. Existing studies present tokens, examples, and hints for corrections, but do not directly explain the reasons in natural language. Although methods that use Large Language Models (LLMs) to provide direct explanations in natural language have been proposed for various tasks, no such method exists for GEC. Generating explanations for GEC corrections involves aligning input and output tokens, identifying correction points, and presenting corresponding explanations consistently. However, it is not straightforward to specify a complex format to generate explanations, because explicit control of generation is difficult with prompts. This study introduces a method called controlled generation with Prompt Insertion (PI) so that LLMs can explain the reasons for corrections in natural language. In PI, LLMs first correct the input text, and then we automatically extract the correction points based on the rules. The extracted correction points are sequentially inserted into the LLM’s explanation output as prompts, guiding the LLMs to generate explanations for the correction points. We also create an Explainable GEC (XGEC) dataset of correction reasons by annotating NUCLE, CoNLL2013, and CoNLL2014. Although generations from GPT-3.5 and ChatGPT using original prompts miss some correction points, the generation control using PI can explicitly guide to describe explanations for all correction points, contributing to improved performance in generating correction reasons.
    </details>
+ **Evaluating Prompting Strategies for Grammatical Error Correction Based on Language Proficiency**
  + Authors: Min Zeng, Jiexin Kuang, Mengyang Qiu, Jayoung Song, Jungyeul Park
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.569/
  + <details>
      <summary>Abstract</summary>
      This paper proposes an analysis of prompting strategies for grammatical error correction (GEC) with selected large language models (LLM) based on language proficiency. GEC using generative LLMs has been known for overcorrection where results obtain higher recall measures than precision measures. The writing examples of English language learners may be different from those of native speakers. Given that there is a significant differences in second language (L2) learners’ error types by their proficiency levels, this paper attempts to reduce overcorrection by examining the interaction between LLM’s performance and L2 language proficiency. Our method focuses on zero-shot and few-shot prompting and fine-tuning models for GEC for learners of English as a foreign language based on the different proficiency. We investigate GEC results and find that overcorrection happens primarily in advanced language learners’ writing (proficiency C) rather than proficiency A (a beginner level) and proficiency B (an intermediate level). Fine-tuned LLMs, and even few-shot prompting with writing examples of English learners, actually tend to exhibit decreased recall measures. To make our claim concrete, we conduct a comprehensive examination of GEC outcomes and their evaluation results based on language proficiency.
    </details>
+ **Evaluation of Really Good Grammatical Error Correction**
  + Authors: Robert Östling, Katarina Gillholm, Murathan Kurfalı, Marie Mattson, Mats Wirén
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.584/
  + <details>
      <summary>Abstract</summary>
      Traditional evaluation methods for Grammatical Error Correction (GEC) fail to fully capture the full range of system capabilities and objectives. The emergence of large language models (LLMs) has further highlighted the shortcomings of these evaluation strategies, emphasizing the need for a paradigm shift in evaluation methodology. In the current study, we perform a comprehensive evaluation of various GEC systems using a recently published dataset of Swedish learner texts. The evaluation is performed using established evaluation metrics as well as human judges. We find that GPT-3 in a few-shot setting by far outperforms previous grammatical error correction systems for Swedish, a language comprising only about 0.1% of its training data. We also found that current evaluation methods contain undesirable biases that a human evaluation is able to reveal. We suggest using human post-editing of GEC system outputs to analyze the amount of change required to reach native-level human performance on the task, and provide a dataset annotated with human post-edits and assessments of grammaticality, fluency and meaning preservation of GEC system outputs.
    </details>
+ **GPT-3.5 for Grammatical Error Correction**
  + Authors: Anisia Katinskaia, Roman Yangarber
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.692/
  + <details>
      <summary>Abstract</summary>
      This paper investigates the application of GPT-3.5 for Grammatical Error Correction (GEC) in multiple languages in several settings: zero-shot GEC, fine-tuning for GEC, and using GPT-3.5 to re-rank correction hypotheses generated by other GEC models. In the zero-shot setting, we conduct automatic evaluations of the corrections proposed by GPT-3.5 using several methods: estimating grammaticality with language models (LMs), the Scribendy test, and comparing the semantic embeddings of sentences. GPT-3.5 has a known tendency to over-correct erroneous sentences and propose alternative corrections. For several languages, such as Czech, German, Russian, Spanish, and Ukrainian, GPT-3.5 substantially alters the source sentences, including their semantics, which presents significant challenges for evaluation with reference-based metrics. For English, GPT-3.5 demonstrates high recall, generates fluent corrections, and generally preserves sentence semantics. However, human evaluation for both English and Russian reveals that, despite its strong error-detection capabilities, GPT-3.5 struggles with several error types, including punctuation mistakes, tense errors, syntactic dependencies between words, and lexical compatibility at the sentence level.
    </details>
+ **Grammatical Error Correction for Code-Switched Sentences by Learners of English**
  + Authors: Kelvin Wey Han Chan, Christopher Bryant, Li Nguyen, Andrew Caines, Zheng Yuan
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.698/
  + <details>
      <summary>Abstract</summary>
      Code-switching (CSW) is a common phenomenon among multilingual speakers where multiple languages are used in a single discourse or utterance. Mixed language utterances may still contain grammatical errors however, yet most existing Grammar Error Correction (GEC) systems have been trained on monolingual data and not developed with CSW in mind. In this work, we conduct the first exploration into the use of GEC systems on CSW text. Through this exploration, we propose a novel method of generating synthetic CSW GEC datasets by translating different spans of text within existing GEC corpora. We then investigate different methods of selecting these spans based on CSW ratio, switch-point factor and linguistic constraints, and identify how they affect the performance of GEC systems on CSW text. Our best model achieves an average increase of 1.57 F0.5 across 3 CSW test sets (English-Chinese, English-Korean and English-Japanese) without affecting the model’s performance on a monolingual dataset. We furthermore discovered that models trained on one CSW language generalise relatively well to other typologically similar CSW languages.
    </details>
+ **Improving Copy-oriented Text Generation via EDU Copy Mechanism**
  + Authors: Tianxiang Wu, Han Chen, Luozheng Qin, Ziqiang Cao, Chunhui Ai
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.768/
  + <details>
      <summary>Abstract</summary>
      Many text generation tasks are copy-oriented. For instance, nearly 30% content of news summaries is copied. The copy rate is even higher in Grammatical Error Correction (GEC). However, existing generative models generate texts through word-by-word decoding, which may lead to factual inconsistencies and slow inference. While Elementary Discourse Units (EDUs) are outstanding extraction units, EDU-based extractive methods can alleviate the aforementioned problems. As a consequence, we propose EDUCopy, a framework that integrates the behavior of copying EDUs into generative models. The main idea of EDUCopy is to use special index tags to represent the copied EDUs during generation. Specifically, we extract important EDUs from input sequences, finetune generative models to generate sequences with special index tags, and restore the generated special index tags into corresponding text spans. By doing so, EDUCopy reduces the number of generated tokens significantly. To verify the effectiveness of EDUCopy, we conduct experiments on the news summarization datasets CNNDM, NYT and the GEC datasets FCE, WI-LOCNESS. While achieving notable ROUGE and M2 scores, GPT-4 evaluation validates the strength of our models in terms of factual consistency, fluency, and overall performance. Moreover, compared to baseline models, EDUCopy achieves a significant acceleration of 1.65x.
    </details>
+ **Improving Grammatical Error Correction by Correction Acceptability Discrimination**
  + Authors: Bin Cao, Kai Jiang, Fayu Pan, Chenlei Bao, Jing Fan
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.772/
  + <details>
      <summary>Abstract</summary>
      Existing Grammatical Error Correction (GEC) methods often overlook the assessment of sentence-level syntax and semantics in the corrected sentence. This oversight results in final corrections that may not be acceptable in the context of the original sentence. In this paper, to improve the performance of Grammatical Error Correction methods, we propose the post-processing task of Correction Acceptability Discrimination (CAD) which aims to remove invalid corrections by comparing the source sentence and its corrected version from the perspective of “sentence-level correctness”. To solve the CAD task, we propose a pipeline method where the acceptability of each possible correction combination based on the predicted corrections for a source sentence will be judged by a discriminator. Within the discriminator, we design a symmetrical comparison operator to overcome the conflicting results that might be caused by the sentence concatenation order. Experiments show that our method can averagely improve F0.5 score by 1.01% over 13 GEC systems in the BEA-2019 test set.
    </details>
+ **LM-Combiner: A Contextual Rewriting Model for Chinese Grammatical Error Correction**
  + Authors: Yixuan Wang, Baoxin Wang, Yijun Liu, Dayong Wu, Wanxiang Che
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.934/
  + <details>
      <summary>Abstract</summary>
      Over-correction is a critical problem in Chinese grammatical error correction (CGEC) task. Recent work using model ensemble methods based on voting can effectively mitigate over-correction and improve the precision of the GEC system. However, these methods still require the output of several GEC systems and inevitably lead to reduced error recall. In this light, we propose the LM-Combiner, a rewriting model that can directly modify the over-correction of GEC system outputs without a model ensemble. Specifically, we train the model on an over-correction dataset constructed through the proposed K-fold cross inference method, which allows it to directly generate filtered sentences by combining the original and the over-corrected text. In the inference stage, we directly take the original sentences and the output results of other systems as input and then obtain the filtered sentences through LM-Combiner. Experiments on the FCGEC dataset show that our proposed method effectively alleviates the over-correction of the original system (+18.2 Precision) while ensuring the error recall remains unchanged. Besides, we find that LM-Combiner still has a good rewriting performance even with small parameters and few training data, and thus can cost-effectively mitigate the over-correction of black-box GEC systems (e.g., ChatGPT).
    </details>
+ **Logging Keystrokes in Writing by English Learners**
  + Authors: Georgios Velentzas, Andrew Caines, Rita Borgo, Erin Pacquetet, Clive Hamilton, Taylor Arnold, Diane Nicholls, Paula Buttery, Thomas Gaillat, Nicolas Ballier, Helen Yannakoudakis
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.938/
  + <details>
      <summary>Abstract</summary>
      Essay writing is a skill commonly taught and practised in schools. The ability to write a fluent and persuasive essay is often a major component of formal assessment. In natural language processing and education technology we may work with essays in their final form, for example to carry out automated assessment or grammatical error correction. In this work we collect and analyse data representing the essay writing process from start to finish, by recording every key stroke from multiple writers participating in our study. We describe our data collection methodology, the characteristics of the resulting dataset, and the assignment of proficiency levels to the texts. We discuss the ways the keystroke data can be used – for instance seeking to identify patterns in the keystrokes which might act as features in automated assessment or may enable further advancements in writing assistance – and the writing support technology which could be built with such information, if we can detect when writers are struggling to compose a section of their essay and offer appropriate intervention. We frame this work in the context of English language learning, but we note that keystroke logging is relevant more broadly to text authoring scenarios as well as cognitive or linguistic analyses of the writing process.
    </details>
+ **Spivavtor: An Instruction Tuned Ukrainian Text Editing Model**
  + Authors: Aman Saini, Artem Chernodub, Vipul Raheja, Vivek Kulkarni
  + Conference: COLING
  + Link: https://aclanthology.org/2024.unlp-1.12/
  + <details>
      <summary>Abstract</summary>
      We introduce Spivavtor, a dataset, and instruction-tuned models for text editing focused on the Ukrainian language. Spivavtor is the Ukrainian-focused adaptation of the English-only CoEdIT (Raheja et al., 2023) model. Similar to CoEdIT, Spivavtor performs text editing tasks by following instructions in Ukrainian like “Виправте граматику в цьому реченнi” and “Спростiть це речення” which translate to “Correct the grammar in this sentence” and “Simplify this sentence” in English, respectively. This paper describes the details of the Spivavtor-Instruct dataset and Spivavtor models. We evaluate Spivavtor on a variety of text editing tasks in Ukrainian, such as Grammatical Error Correction (GEC), Text Simplification, Coherence, and Paraphrasing, and demonstrate its superior performance on all of them. We publicly release our best performing models and data as resources to the community to advance further research in this space.
    </details>
+ **Universal Dependencies for Learner Russian**
  + Authors: Alla Rozovskaya
  + Conference: COLING
  + Link: https://aclanthology.org/2024.lrec-main.1486/
  + <details>
      <summary>Abstract</summary>
      We introduce a pilot annotation of Russian learner data with syntactic dependency relations. The annotation is performed on a subset of sentences from RULEC-GEC and RU-Lang8, two error-corrected Russian learner datasets. We provide manually labeled Universal Dependency (UD) trees for 500 sentence pairs, annotating both the original (source) and the corrected (target) version of each sentence. Further, we outline guidelines for annotating learner Russian data containing non-standard erroneous text and analyze the effect that the individual errors have on the resulting dependency trees. This study should contribute to a wide range of computational and theoretical research directions in second language learning and grammatical error correction.
    </details>
+ **Multi-Reference Benchmarks for Russian Grammatical Error Correction**
  + Authors: Frank Palma Gomez, Alla Rozovskaya
  + Conference: EACL
  + Link: https://aclanthology.org/2024.eacl-long.76/
  + <details>
      <summary>Abstract</summary>
      This paper presents multi-reference benchmarks for the Grammatical Error Correction (GEC) of Russian, based on two existing single-reference datasets, for a total of 7,444 learner sentences from a variety of first language backgrounds. Each sentence is corrected independently by two new raters, and their corrections are reviewed by a senior annotator, resulting in a total of three references per sentence. Analysis of the annotations reveals that the new raters tend to make more changes, compared to the original raters, especially at the lexical level. We conduct experiments with two popular GEC approaches and show competitive performance on the original datasets and the new benchmarks. We also compare system scores as evaluated against individual annotators and discuss the effect of using multiple references overall and on specific error types. We find that using the union of the references increases system scores by more than 10 points and decreases the gap between system and human performance, thereby providing a more realistic evaluation of GEC system performance, although the effect is not the same across the error types. The annotations are available for research.
    </details>
+ **No Error Left Behind: Multilingual Grammatical Error Correction with Pre-trained Translation Models**
  + Authors: Agnes Luhtaru, Elizaveta Korotkova, Mark Fishel
  + Conference: EACL
  + Link: https://aclanthology.org/2024.eacl-long.73/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) enhances language proficiency and promotes effective communication, but research has primarily centered around English. We propose a simple approach to multilingual and low-resource GEC by exploring the potential of multilingual machine translation (MT) models for error correction. We show that MT models are not only capable of error correction out-of-the-box, but that they can also be fine-tuned to even better correction quality. Results show the effectiveness of this approach, with our multilingual model outperforming similar-sized mT5-based models and even competing favourably with larger models.
    </details>
## 2023
+ **A Closer Look at k-Nearest Neighbors Grammatical Error Correction**
  + Authors: Justin Vasselli, Taro Watanabe
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.19/
  + <details>
      <summary>Abstract</summary>
      In various natural language processing tasks, such as named entity recognition and machine translation, example-based approaches have been used to improve performance by leveraging existing knowledge. However, the effectiveness of this approach for Grammatical Error Correction (GEC) is unclear. In this work, we explore how an example-based approach affects the accuracy and interpretability of the output of GEC systems and the trade-offs involved. The approach we investigate has shown great promise in machine translation by using the $k$-nearest translation examples to improve the results of a pretrained Transformer model. We find that using this technique increases precision by reducing the number of false positives, but recall suffers as the model becomes more conservative overall. Increasing the number of example sentences in the datastore does lead to better performing systems, but with diminishing returns and a high decoding cost. Synthetic data can be used as examples, but the effectiveness varies depending on the base model. Finally, we find that finetuning on a set of data may be more effective than using that data during decoding as examples.
    </details>
+ **Are Pre-trained Language Models Useful for Model Ensemble in Chinese Grammatical Error Correction?**
  + Authors: Chenming Tang, Xiuyu Wu, Yunfang Wu
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-short.77/
  + <details>
      <summary>Abstract</summary>
      Model ensemble has been in widespread use for Grammatical Error Correction (GEC), boosting model performance. We hypothesize that model ensemble based on the perplexity (PPL) computed by pre-trained language models (PLMs) should benefit the GEC system. To this end, we explore several ensemble strategies based on strong PLMs with four sophisticated single models. However, the performance does not improve but even gets worse after the PLM-based ensemble. This surprising result sets us doing a detailed analysis on the data and coming up with some insights on GEC. The human references of correct sentences is far from sufficient in the test data, and the gap between a correct sentence and an idiomatic one is worth our attention. Moreover, the PLM-based ensemble strategies provide an effective way to extend and improve GEC benchmark data. Our source code is available at https://github.com/JamyDon/PLM-based-CGEC-Model-Ensemble.
    </details>
+ **Byte-Level Grammatical Error Correction Using Synthetic and Curated Corpora**
  + Authors: Svanhvít Lilja Ingólfsdóttir, Petur Ragnarsson, Haukur Jónsson, Haukur Simonarson, Vilhjalmur Thorsteinsson, Vésteinn Snæbjarnarson
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.402/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is the task of correcting typos, spelling, punctuation and grammatical issues in text. Approaching the problem as a sequence-to-sequence task, we compare the use of a common subword unit vocabulary and byte-level encoding. Initial synthetic training data is created using an error-generating pipeline, and used for finetuning two subword-level models and one byte-level model. Models are then finetuned further on hand-corrected error corpora, including texts written by children, university students, dyslexic and second-language writers, and evaluated over different error types and error origins. We show that a byte-level model enables higher correction quality than a subword approach, not only for simple spelling errors, but also for more complex semantic, stylistic and grammatical issues. In particular, initial training on synthetic corpora followed by finetuning on a relatively small parallel corpus of real-world errors helps the byte-level model correct a wide range of commonly occurring errors. Our experiments are run for the Icelandic language but should hold for other similar languages, and in particular to morphologically rich ones.
    </details>
+ **Enhancing Grammatical Error Correction Systems with Explanations**
  + Authors: Yuejiao Fei, Leyang Cui, Sen Yang, Wai Lam, Zhenzhong Lan, Shuming Shi
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.413/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction systems improve written communication by detecting and correcting language mistakes. To help language learners better understand why the GEC system makes a certain correction, the causes of errors (evidence words) and the corresponding error types are two key factors. To enhance GEC systems with explanations, we introduce EXPECT, a large dataset annotated with evidence words and grammatical error types. We propose several baselines and anlysis to understand this task. Furthermore, human evaluation verifies our explainable GEC system’s explanations can assist second-language learners in determining whether to accept a correction suggestion and in understanding the associated grammar rule.
    </details>
+ **Exploring Effectiveness of GPT-3 in Grammatical Error Correction: A Study on Performance and Controllability in Prompt-Based Methods**
  + Authors: Mengsay Loem, Masahiro Kaneko, Sho Takase, Naoaki Okazaki
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.18/
  + <details>
      <summary>Abstract</summary>
      Large-scale pre-trained language models such as GPT-3 have shown remarkable performance across various natural language processing tasks. However, applying prompt-based methods with GPT-3 for Grammatical Error Correction (GEC) tasks and their controllability remains underexplored. Controllability in GEC is crucial for real-world applications, particularly in educational settings, where the ability to tailor feedback according to learner levels and specific error types can significantly enhance the learning process. This paper investigates the performance and controllability of prompt-based methods with GPT-3 for GEC tasks using zero-shot and few-shot setting. We explore the impact of task instructions and examples on GPT-3’s output, focusing on controlling aspects such as minimal edits, fluency edits, and learner levels. Our findings demonstrate that GPT-3 could effectively perform GEC tasks, outperforming existing supervised and unsupervised approaches. We also showed that GPT-3 could achieve controllability when appropriate task instructions and examples are given.
    </details>
+ **GEC-DePenD: Non-Autoregressive Grammatical Error Correction with Decoupled Permutation and Decoding**
  + Authors: Konstantin Yakovlev, Alexander Podolskiy, Andrey Bout, Sergey Nikolenko, Irina Piontkovskaya
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.86/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is an important NLP task that is currently usually solved with autoregressive sequence-to-sequence models. However, approaches of this class are inherently slow due to one-by-one token generation, so non-autoregressive alternatives are needed. In this work, we propose a novel non-autoregressive approach to GEC that decouples the architecture into a permutation network that outputs a self-attention weight matrix that can be used in beam search to find the best permutation of input tokens (with auxiliary <ins> tokens) and a decoder network based on a step-unrolled denoising autoencoder that fills in specific tokens. This allows us to find the token permutation after only one forward pass of the permutation network, avoiding autoregressive constructions. We show that the resulting network improves over previously known non-autoregressive methods for GEC and reaches the level of autoregressive methods that do not use language-specific synthetic data generation methods. Our results are supported by a comprehensive experimental validation on the ConLL-2014 and BEA datasets and an extensive ablation study that supports our architectural and algorithmic choices.
    </details>
+ **Gender-Inclusive Grammatical Error Correction through Augmentation**
  + Authors: Gunnar Lund, Kostiantyn Omelianchuk, Igor Samokhin
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.13/
  + <details>
      <summary>Abstract</summary>
      In this paper we show that GEC systems display gender bias related to the use of masculine and feminine terms and the gender-neutral singular “they”. We develop parallel datasets of texts with masculine and feminine terms, and singular “they”, and use them to quantify gender bias in three competitive GEC systems. We contribute a novel data augmentation technique for singular “they” leveraging linguistic insights about its distribution relative to plural “they”. We demonstrate that both this data augmentation technique and a refinement of a similar augmentation technique for masculine and feminine terms can generate training data that reduces bias in GEC systems, especially with respect to singular “they” while maintaining the same level of quality.
    </details>
+ **Grammatical Error Correction for Sentence-level Assessment in Language Learning**
  + Authors: Anisia Katinskaia, Roman Yangarber
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.41/
  + <details>
      <summary>Abstract</summary>
      The paper presents experiments on using a Grammatical Error Correction (GEC) model to assess the correctness of answers that language learners give to grammar exercises. We explored whether a GEC model can be applied in the language learning context for a language with complex morphology. We empirically check a hypothesis that a GEC model corrects only errors and leaves correct answers unchanged. We perform a test on assessing learner answers in a real but constrained language-learning setup: the learners answer only fill-in-the-blank and multiple-choice exercises. For this purpose, we use ReLCo, a publicly available manually annotated learner dataset in Russian (Katinskaia et al., 2022). In this experiment, we fine-tune a large-scale T5 language model for the GEC task and estimate its performance on the RULEC-GEC dataset (Rozovskaya and Roth, 2019) to compare with top-performing models. We also release an updated version of the RULEC-GEC test set, manually checked by native speakers. Our analysis shows that the GEC model performs reasonably well in detecting erroneous answers to grammar exercises and potentially can be used for best-performing error types in a real learning setup. However, it struggles to assess answers which were tagged by human annotators as alternative-correct using the aforementioned hypothesis. This is in large part due to a still low recall in correcting errors, and the fact that the GEC model may modify even correct words—it may generate plausible alternatives, which are hard to evaluate against the gold-standard reference.
    </details>
+ **NetEase.AI at SemEval-2023 Task 2: Enhancing Complex Named Entities Recognition in Noisy Scenarios via Text Error Correction and External Knowledge**
  + Authors: Ruixuan Lu, Zihang Tang, Guanglong Hu, Dong Liu, Jiacheng Li
  + Conference: ACL
  + Link: https://aclanthology.org/2023.semeval-1.124/
  + <details>
      <summary>Abstract</summary>
      Complex named entities (NE), like the titles of creative works, are not simple nouns and pose challenges for NER systems. In the SemEval 2023, Task 2: MultiCoNER II was proposed, whose goal is to recognize complex entities against out of knowledge-base entities and noisy scenarios. To address the challenges posed by MultiCoNER II, our team NetEase.AI proposed an entity recognition system that integrates text error correction system and external knowledge, which can recognize entities in scenes that contain entities out of knowledge base and text with noise. Upon receiving an input sentence, our systems will correct the sentence, extract the entities in the sentence as candidate set using the entity recognition model that incorporates the gazetteer information, and then use the external knowledge to classify the candidate entities to obtain entity type features. Finally, our system fused the multi-dimensional features of the candidate entities into a stacking model, which was used to select the correct entities from the candidate set as the final output. Our system exhibited good noise resistance and excellent entity recognition performance, resulting in our team’s first place victory in the Chinese track of MultiCoNER II.
    </details>
+ **PEEP-Talk: A Situational Dialogue-based Chatbot for English Education**
  + Authors: Seungjun Lee, Yoonna Jang, Chanjun Park, Jungseob Lee, Jaehyung Seo, Hyeonseok Moon, Sugyeong Eo, Seounghoon Lee, Bernardo Yahya, Heuiseok Lim
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-demo.18/
  + <details>
      <summary>Abstract</summary>
      English is acknowledged worldwide as a mode of communication. However, due to the absence of realistic practicing scenarios, students learning English as a foreign language (EFL) typically have limited chances to converse and share feedback with others. In this paper, we propose PEEP-Talk, a real-world situational dialogue-based chatbot designed for English education. It also naturally switches to a new topic or situation in response to out-of-topic utterances, which are common among English beginners. Furthermore, PEEP-Talk provides feedback score on conversation and grammar error correction. We performed automatic and user evaluations to validate performance and education efficiency of our system. The results show that PEEP-Talk generates appropriate responses in various real-life situations while providing accurate feedback to learners. Moreover, we demonstrate a positive impact on English-speaking, grammar, and English learning anxiety, implying that PEEP-Talk can lower the barrier to learning natural conversation in effective ways.
    </details>
+ **TemplateGEC: Improving Grammatical Error Correction with Detection Template**
  + Authors: Yinghao Li, Xuebo Liu, Shuo Wang, Peiyuan Gong, Derek F. Wong, Yang Gao, Heyan Huang, Min Zhang
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.380/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) can be divided into sequence-to-edit (Seq2Edit) and sequence-to-sequence (Seq2Seq) frameworks, both of which have their pros and cons. To utilize the strengths and make up for the shortcomings of these frameworks, this paper proposes a novel method, TemplateGEC, which capitalizes on the capabilities of both Seq2Edit and Seq2Seq frameworks in error detection and correction respectively. TemplateGEC utilizes the detection labels from a Seq2Edit model, to construct the template as the input. A Seq2Seq model is employed to enforce consistency between the predictions of different templates by utilizing consistency learning. Experimental results on the Chinese NLPCC18, English BEA19 and CoNLL14 benchmarks show the effectiveness and robustness of TemplateGEC.Further analysis reveals the potential of our method in performing human-in-the-loop GEC. Source code and scripts are available at https://github.com/li-aolong/TemplateGEC.
    </details>
+ **Token-Level Self-Evolution Training for Sequence-to-Sequence Learning**
  + Authors: Keqin Peng, Liang Ding, Qihuang Zhong, Yuanxin Ouyang, Wenge Rong, Zhang Xiong, Dacheng Tao
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-short.73/
  + <details>
      <summary>Abstract</summary>
      Adaptive training approaches, widely used in sequence-to-sequence models, commonly reweigh the losses of different target tokens based on priors, e.g. word frequency. However, most of them do not consider the variation of learning difficulty in different training steps, and overly emphasize the learning of difficult one-hot labels, making the learning deterministic and sub-optimal. In response, we present Token-Level Self-Evolution Training (SE), a simple and effective dynamic training method to fully and wisely exploit the knowledge from data. SE focuses on dynamically learning the under-explored tokens for each forward pass and adaptively regularizes the training by introducing a novel token-specific label smoothing approach. Empirically, SE yields consistent and significant improvements in three tasks, i.e. machine translation, summarization, and grammatical error correction. Encouragingly, we achieve averaging +0.93 BLEU improvement on three machine translation tasks. Analyses confirm that, besides improving lexical accuracy, SE enhances generation diversity and model generalization.
    </details>
+ **Towards standardizing Korean Grammatical Error Correction: Datasets and Annotation**
  + Authors: Soyoung Yoon, Sungjoon Park, Gyuwan Kim, Junhee Cho, Kihyo Park, Gyu Tae Kim, Minjoon Seo, Alice Oh
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.371/
  + <details>
      <summary>Abstract</summary>
      Research on Korean grammatical error correction (GEC) is limited, compared to other major languages such as English. We attribute this problematic circumstance to the lack of a carefully designed evaluation benchmark for Korean GEC. In this work, we collect three datasets from different sources (Kor-Lang8, Kor-Native, and Kor-Learner) that covers a wide range of Korean grammatical errors. Considering the nature of Korean grammar, We then define 14 error types for Korean and provide KAGAS (Korean Automatic Grammatical error Annotation System), which can automatically annotate error types from parallel corpora. We use KAGAS on our datasets to make an evaluation benchmark for Korean, and present baseline models trained from our datasets. We show that the model trained with our datasets significantly outperforms the currently used statistical Korean GEC system (Hanspell) on a wider range of error types, demonstrating the diversity and usefulness of the datasets. The implementations and datasets are open-sourced.
    </details>
+ **Training for Grammatical Error Correction Without Human-Annotated L2 Learners’ Corpora**
  + Authors: Mikio Oda
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.38/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is a challenging task for non-native second language (L2) learners and learning machines. Data-driven GEC learning requires as much human-annotated genuine training data as possible. However, it is difficult to produce larger-scale human-annotated data, and synthetically generated large-scale parallel training data is valuable for GEC systems. In this paper, we propose a method for rebuilding a corpus of synthetic parallel data using target sentences predicted by a GEC model to improve performance. Experimental results show that our proposed pre-training outperforms that on the original synthetic datasets. Moreover, it is also shown that our proposed training without human-annotated L2 learners’ corpora is as practical as conventional full pipeline training with both synthetic datasets and L2 learners’ corpora in terms of accuracy.
    </details>
+ **Bidirectional Transformer Reranker for Grammatical Error Correction**
  + Authors: Ying Zhang, Hidetaka Kamigaito, Manabu Okumura
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.234/
  + <details>
      <summary>Abstract</summary>
      Pre-trained seq2seq models have achieved state-of-the-art results in the grammatical error correction task. However, these models still suffer from a prediction bias due to their unidirectional decoding. Thus, we propose a bidirectional Transformer reranker (BTR), that re-estimates the probability of each candidate sentence generated by the pre-trained seq2seq model. The BTR preserves the seq2seq-style Transformer architecture but utilizes a BERT-style self-attention mechanism in the decoder to compute the probability of each target token by using masked language modeling to capture bidirectional representations from the target context. For guiding the reranking, the BTR adopts negative sampling in the objective function to minimize the unlikelihood. During inference, the BTR gives final results after comparing the reranked top-1 results with the original ones by an acceptance threshold. Experimental results show that, in reranking candidates from a pre-trained seq2seq model, T5-base, the BTR on top of T5-base could yield 65.47 and 71.27 F0.5 scores on the CoNLL-14 and BEA test sets, respectively, and yield 59.52 GLEU score on the JFLEG corpus, with improvements of 0.36, 0.76 and 0.48 points compared with the original T5-base. Furthermore, when reranking candidates from T5-large, the BTR on top of T5-base improved the original T5-large by 0.26 points on the BEA test set.
    </details>
+ **Focal Training and Tagger Decouple for Grammatical Error Correction**
  + Authors: Minghuan Tan, Min Yang, Ruifeng Xu
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.370/
  + <details>
      <summary>Abstract</summary>
      In this paper, we investigate how to improve tagging-based Grammatical Error Correction models. We address two issues of current tagging-based approaches, label imbalance issue, and tagging entanglement issue. Then we propose to down-weight the loss of well-classified labels using Focal Loss and decouple the error detection layer from the label tagging layer through an extra self-attention-based matching module. Experiments over three latest Chinese Grammatical Error Correction datasets show that our proposed methods are effective. We further analyze choices of hyper-parameters for Focal Loss and inference tweaking.
    </details>
+ **Improving Autoregressive Grammatical Error Correction with Non-autoregressive Models**
  + Authors: Hang Cao, Zhiquan Cao, Chi Hu, Baoyu Hou, Tong Xiao, Jingbo Zhu
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.760/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) aims to correct grammatical errors in sentences. We find that autoregressive models tend to assign low probabilities to tokens that need corrections. Here we introduce additional signals to the training of GEC models so that these systems can learn to better predict at ambiguous positions. To do this, we use a non-autoregressive model as an auxiliary model, and develop a new regularization term of training by considering the difference in predictions between the autoregressive and non-autoregressive models. We experiment with this method on both English and Chinese GEC tasks. Experimental results show that our GEC system outperforms the baselines on all the data sets significantly.
    </details>
+ **Improving Grammatical Error Correction with Multimodal Feature Integration**
  + Authors: Tao Fang, Jinpeng Hu, Derek F. Wong, Xiang Wan, Lidia S. Chao, Tsung-Hui Chang
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.594/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is a promising task aimed at correcting errors in a text. Many methods have been proposed to facilitate this task with remarkable results. However, most of them only focus on enhancing textual feature extraction without exploring the usage of other modalities’ information (e.g., speech), which can also provide valuable knowledge to help the model detect grammatical errors. To shore up this deficiency, we propose a novel framework that integrates both speech and text features to enhance GEC. In detail, we create new multimodal GEC datasets for English and German by generating audio from text using the advanced text-to-speech models. Subsequently, we extract acoustic and textual representations by a multimodal encoder that consists of a speech and a text encoder. A mixture-of-experts (MoE) layer is employed to selectively align representations from the two modalities, and then a dot attention mechanism is used to fuse them as final multimodal representations. Experimental results on CoNLL14, BEA19 English, and Falko-MERLIN German show that our multimodal GEC models achieve significant improvements over strong baselines and achieve a new state-of-the-art result on the Falko-MERLIN test set.
    </details>
+ **LET: Leveraging Error Type Information for Grammatical Error Correction**
  + Authors: Lingyu Yang, Hongjia Li, Lei Li, Chengyin Xu, Shutao Xia, Chun Yuan
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.371/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) aims to correct errors in given sentences and is significant to many downstream natural language understanding tasks. Recent work introduces the idea of grammatical error detection (GED) to improve the GEC task performance. In contrast, these explicit multi-stage works propagate and amplify the problem of misclassification of the GED module. To introduce more convincing error type information, we propose an end-to-end framework in this paper, which Leverages Error Type (LET) information in the generation process. First, the input text is fed into a classification module to obtain the error type corresponding to each token. Then, we introduce the category information into the decoder’s input and cross-attention module in two ways, respectively. Experiments on various datasets show that our proposed method outperforms existing methods by a clear margin.
    </details>
+ **Leveraging Denoised Abstract Meaning Representation for Grammatical Error Correction**
  + Authors: Hejing Cao, Dongyan Zhao
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.449/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is the task of correcting errorful sentences into grammatically correct, semantically consistent, and coherent sentences. Popular GEC models either use large-scale synthetic corpora or use a large number of human-designed rules. The former is costly to train, while the latter requires quite a lot of human expertise. In recent years, AMR, a semantic representation framework, has been widely used by many natural language tasks due to its completeness and flexibility. A non-negligible concern is that AMRs of grammatically incorrect sentences may not be exactly reliable. In this paper, we propose the AMR-GEC, a seq-to-seq model that incorporates denoised AMR as additional knowledge. Specifically, We design a semantic aggregated GEC model and explore denoising methods to get AMRs more reliable. Experiments on the BEA-2019 shared task and the CoNLL-2014 shared task have shown that AMR-GEC performs comparably to a set of strong baselines with a large number of synthetic data. Compared with the T5 model with synthetic data, AMR-GEC can reduce the training time by 32% while inference time is comparable. To the best of our knowledge, we are the first to incorporate AMR for grammatical error correction.
    </details>
+ **NaSGEC: a Multi-Domain Chinese Grammatical Error Correction Dataset from Native Speaker Texts**
  + Authors: Yue Zhang, Bo Zhang, Haochen Jiang, Zhenghua Li, Chen Li, Fei Huang, Min Zhang
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.630/
  + <details>
      <summary>Abstract</summary>
      We introduce NaSGEC, a new dataset to facilitate research on Chinese grammatical error correction (CGEC) for native speaker texts from multiple domains. Previous CGEC research primarily focuses on correcting texts from a single domain, especially learner essays. To broaden the target domain, we annotate multiple references for 12,500 sentences from three native domains, i.e., social media, scientific writing, and examination. We provide solid benchmark results for NaSGEC by employing cutting-edge CGEC models and different training data. We further perform detailed analyses of the connections and gaps between our domains from both empirical and statistical views. We hope this work can inspire future studies on an important but under-explored direction–cross-domain GEC.
    </details>
+ **TransGEC: Improving Grammatical Error Correction with Translationese**
  + Authors: Tao Fang, Xuebo Liu, Derek F. Wong, Runzhe Zhan, Liang Ding, Lidia S. Chao, Dacheng Tao, Min Zhang
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.223/
  + <details>
      <summary>Abstract</summary>
      Data augmentation is an effective way to improve model performance of grammatical error correction (GEC). This paper identifies a critical side-effect of GEC data augmentation, which is due to the style discrepancy between the data used in GEC tasks (i.e., texts produced by non-native speakers) and data augmentation (i.e., native texts). To alleviate this issue, we propose to use an alternative data source, translationese (i.e., human-translated texts), as input for GEC data augmentation, which 1) is easier to obtain and usually has better quality than non-native texts, and 2) has a more similar style to non-native texts. Experimental results on the CoNLL14 and BEA19 English, NLPCC18 Chinese, Falko-MERLIN German, and RULEC-GEC Russian GEC benchmarks show that our approach consistently improves correction accuracy over strong baselines. Further analyses reveal that our approach is helpful for overcoming mainstream correction difficulties such as the corrections of frequent words, missing words, and substitution errors. Data, code, models and scripts are freely available at https://github.com/NLP2CT/TransGEC.
    </details>
+ **Advancements in Arabic Grammatical Error Detection and Correction: An Empirical Investigation**
  + Authors: Bashar Alhafni, Go Inoue, Christian Khairallah, Nizar Habash
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.396/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is a well-explored problem in English with many existing models and datasets. However, research on GEC in morphologically rich languages has been limited due to challenges such as data scarcity and language complexity. In this paper, we present the first results on Arabic GEC using two newly developed Transformer-based pretrained sequence-to-sequence models. We also define the task of multi-class Arabic grammatical error detection (GED) and present the first results on multi-class Arabic GED. We show that using GED information as auxiliary input in GEC models improves GEC performance across three datasets spanning different genres. Moreover, we also investigate the use of contextual morphological preprocessing in aiding GEC systems. Our models achieve SOTA results on two Arabic GEC shared task datasets and establish a strong benchmark on a recently created dataset. We make our code, data, and pretrained models publicly available.
    </details>
+ **Beyond English: Evaluating LLMs for Arabic Grammatical Error Correction**
  + Authors: Sang Kwon, Gagan Bhatia, El Moatez Billah Nagoudi, Muhammad Abdul-Mageed
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.arabicnlp-1.9/
  + <details>
      <summary>Abstract</summary>
      Large language models (LLMs) finetuned to follow human instruction have recently exhibited significant capabilities in various English NLP tasks. However, their performance in grammatical error correction (GEC), especially on languages other than English, remains significantly unexplored. In this work, we evaluate the abilities of instruction finetuned LLMs in Arabic GEC, a complex task due to Arabic’s rich morphology. Our findings suggest that various prompting methods, coupled with (in-context) few-shot learning, demonstrate considerable effectiveness, with GPT-4 achieving up to 65.49 F1 score under expert prompting (approximately 5 points higher than our established baseline). Despite these positive results, we find that instruction finetuned models, regardless of their size, are still outperformed by fully finetuned ones, even if they are significantly smaller in size. This disparity highlights substantial room for improvements for LLMs. Inspired by methods used in low-resource machine translation, we also develop a method exploiting synthetic data that significantly outperforms previous models on two standard Arabic benchmarks. Our best model achieves a new SOTA on Arabic GEC, with 73.29 and 73.26 F1 on the 2014 and 2015 QALB datasets, respectively, compared to peer-reviewed published baselines.
    </details>
+ **CLEME: Debiasing Multi-reference Evaluation for Grammatical Error Correction**
  + Authors: Jingheng Ye, Yinghui Li, Qingyu Zhou, Yangning Li, Shirong Ma, Hai-Tao Zheng, Ying Shen
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.378/
  + <details>
      <summary>Abstract</summary>
      Evaluating the performance of Grammatical Error Correction (GEC) systems is a challenging task due to its subjectivity. Designing an evaluation metric that is as objective as possible is crucial to the development of GEC task. However, mainstream evaluation metrics, i.e., reference-based metrics, introduce bias into the multi-reference evaluation by extracting edits without considering the presence of multiple references. To overcome this issue, we propose Chunk-LE Multi-reference Evaluation (CLEME), designed to evaluate GEC systems in the multi-reference evaluation setting. CLEME builds chunk sequences with consistent boundaries for the source, the hypothesis and references, thus eliminating the bias caused by inconsistent edit boundaries. Furthermore, we observe the consistent boundary could also act as the boundary of grammatical errors, based on which the F0.5 score is then computed following the correction independence assumption. We conduct experiments on six English reference sets based on the CoNLL-2014 shared task. Extensive experiments and detailed analyses demonstrate the correctness of our discovery and the effectiveness of CLEME. Further analysis reveals that CLEME is robust to evaluate GEC systems across reference sets with varying numbers of references and annotation styles. All the source codes of CLEME are released at https://github.com/THUKElab/CLEME.
    </details>
+ **Doolittle: Benchmarks and Corpora for Academic Writing Formalization**
  + Authors: Shizhe Diao, Yongyu Lei, Liangming Pan, Tianqing Fang, Wangchunshu Zhou, Sedrick Keh, Min-Yen Kan, Tong Zhang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.809/
  + <details>
      <summary>Abstract</summary>
      Improving the quality of academic writing is a meaningful but challenging task. Conventional methods of language refinement focus on narrow, specific linguistic features within isolated sentences, such as grammatical errors and improper word use. We propose a more general task, Academic Writing Formalization (AWF), to improve the overall quality of formal academic writing at the paragraph level. We formulate this language refinement task as a formal text style transfer task which transfers informal-academic text to formal-academic and contribute a large-scale non-parallel dataset, Doolittle, for this purpose. Concurrently, we apply a method named metric-oriented reinforcement learning (MORL) to two large language models (LLM) where we incorporate different levels of automatic feedback into the training process. Our experiments reveal that existing text transfer models and grammatical error correction models address certain aspects of AWF but still have a significant performance gap compared to human performance. Meanwhile, language models fine-tuned with our MORL method exhibit considerably improved performance, rivaling the latest chatbot ChatGPT, but still have a non-negligible gap compared to the ground truth formal-academic texts in Doolittle.
    </details>
+ **Efficient Grammatical Error Correction Via Multi-Task Training and Optimized Training Schedule**
  + Authors: Andrey Bout, Alexander Podolskiy, Sergey Nikolenko, Irina Piontkovskaya
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.355/
  + <details>
      <summary>Abstract</summary>
      Progress in neural grammatical error correction (GEC) is hindered by the lack of annotated training data. Sufficient amounts of high-quality manually annotated data are not available, so recent research has relied on generating synthetic data, pretraining on it, and then fine-tuning on real datasets; performance gains have been achieved either by ensembling or by using huge pretrained models such as XXL-T5 as the backbone. In this work, we explore an orthogonal direction: how to use available data more efficiently. First, we propose auxiliary tasks that exploit the alignment between the original and corrected sentences, such as predicting a sequence of corrections. We formulate each task as a sequence-to-sequence problem and perform multi-task training. Second, we discover that the order of datasets used for training and even individual instances within a dataset may have important effects on the final performance, so we set out to find the best training schedule. Together, these two ideas lead to significant improvements, producing results that improve state of the art with much smaller models; in particular, we outperform the best models based on T5-XXL (11B parameters) with a BART-based model (400M parameters).
    </details>
+ **Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks**
  + Authors: Andrea Sottana, Bin Liang, Kai Zou, Zheng Yuan
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.543/
  + <details>
      <summary>Abstract</summary>
      Large Language Models (LLMs) evaluation is a patchy and inconsistent landscape, and it is becoming clear that the quality of automatic evaluation metrics is not keeping up with the pace of development of generative models. We aim to improve the understanding of current models’ performance by providing a preliminary and hybrid evaluation on a range of open and closed-source generative LLMs on three NLP benchmarks: text summarisation, text simplification and grammatical error correction (GEC), using both automatic and human evaluation. We also explore the potential of the recently released GPT-4 to act as an evaluator. We find that ChatGPT consistently outperforms many other popular models according to human reviewers on the majority of metrics, while scoring much more poorly when using classic automatic evaluation metrics. We also find that human reviewers rate the gold reference as much worse than the best models’ outputs, indicating the poor quality of many popular benchmarks. Finally, we find that GPT-4 is capable of ranking models’ outputs in a way which aligns reasonably closely to human judgement despite task-specific variations, with a lower alignment in the GEC task.
    </details>
+ **Reducing Sequence Length by Predicting Edit Spans with Large Language Models**
  + Authors: Masahiro Kaneko, Naoaki Okazaki
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.619/
  + <details>
      <summary>Abstract</summary>
      Large Language Models (LLMs) have demonstrated remarkable performance in various tasks and gained significant attention. LLMs are also used for local sequence transduction tasks, including grammatical error correction (GEC) and formality style transfer, where most tokens in a source text are kept unchanged. However, the models that generate all target tokens in such tasks have a tendency to simply copy the input text as is, without making needed changes, because the difference between input and output texts is minimal in the training data. This is also inefficient because the computational cost grows quadratically with the target sequence length with Transformer. This paper proposes predicting edit spans for the source text for local sequence transduction tasks. Representing an edit span with a position of the source text and corrected tokens, we can reduce the length of the target sequence and the computational cost for inference. We apply instruction tuning for LLMs on the supervision data of edit spans. Experiments show that the proposed method achieves comparable performance to the baseline in four tasks, paraphrasing, formality style transfer, GEC, and text simplification, despite reducing the length of the target text by as small as 21%. Furthermore, we report that the task-specific fine-tuning with the proposed method achieved state-of-the-art performance in the four tasks.
    </details>
+ **RobustGEC: Robust Grammatical Error Correction Against Subtle Context Perturbation**
  + Authors: Yue Zhang, Leyang Cui, Enbo Zhao, Wei Bi, Shuming Shi
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.1043/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) systems play a vital role in assisting people with their daily writing tasks. However, users may sometimes come across a GEC system that initially performs well but fails to correct errors when the inputs are slightly modified. To ensure an ideal user experience, a reliable GEC system should have the ability to provide consistent and accurate suggestions when encountering irrelevant context perturbations, which we refer to as context robustness. In this paper, we introduce RobustGEC, a benchmark designed to evaluate the context robustness of GEC systems. RobustGEC comprises 5,000 GEC cases, each with one original error-correct sentence pair and five variants carefully devised by human annotators. Utilizing RobustGEC, we reveal that state-of-the-art GEC systems still lack sufficient robustness against context perturbations. Moreover, we propose a simple yet effective method for remitting this issue.
    </details>
+ **System Combination via Quality Estimation for Grammatical Error Correction**
  + Authors: Muhammad Reza Qorib, Hwee Tou Ng
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.785/
  + <details>
      <summary>Abstract</summary>
      Quality estimation models have been developed to assess the corrections made by grammatical error correction (GEC) models when the reference or gold-standard corrections are not available. An ideal quality estimator can be utilized to combine the outputs of multiple GEC systems by choosing the best subset of edits from the union of all edits proposed by the GEC base systems. However, we found that existing GEC quality estimation models are not good enough in differentiating good corrections from bad ones, resulting in a low F0.5 score when used for system combination. In this paper, we propose GRECO, a new state-of-the-art quality estimation model that gives a better estimate of the quality of a corrected sentence, as indicated by having a higher correlation to the F0.5 score of a corrected sentence. It results in a combined GEC system with a higher F0.5 score. We also propose three methods for utilizing GEC quality estimation models for system combination with varying generality: model-agnostic, model-agnostic with voting bias, and model-dependent method. The combined GEC system outperforms the state of the art on the CoNLL-2014 test set and the BEA-2019 test set, achieving the highest F0.5 scores published to date.
    </details>
+ **TLM: Token-Level Masking for Transformers**
  + Authors: Yangjun Wu, Kebin Fang, Dongxiang Zhang, Han Wang, Hao Zhang, Gang Chen
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.871/
  + <details>
      <summary>Abstract</summary>
      Structured dropout approaches, such as attention dropout and DropHead, have been investigated to regularize the multi-head attention mechanism in Transformers. In this paper, we propose a new regularization scheme based on token-level rather than structure-level to reduce overfitting. Specifically, we devise a novel Token-Level Masking (TLM) training strategy for Transformers to regularize the connections of self-attention, which consists of two masking techniques that are effective and easy to implement. The underlying idea is to manipulate the connections between tokens in the multi-head attention via masking, where the networks are forced to exploit partial neighbors’ information to produce a meaningful representation. The generality and effectiveness of TLM are thoroughly evaluated via extensive experiments on 4 diversified NLP tasks across 18 datasets, including natural language understanding benchmark GLUE, ChineseGLUE, Chinese Grammatical Error Correction, and data-to-text generation. The results indicate that TLM can consistently outperform attention dropout and DropHead, e.g., it increases by 0.5 points relative to DropHead with BERT-large on GLUE. Moreover, TLM can establish a new record on the data-to-text benchmark Rotowire (18.93 BLEU). Our code will be publicly available at https://github.com/Young1993/tlm.
    </details>
+ **Unsupervised Grammatical Error Correction Rivaling Supervised Methods**
  + Authors: Hannan Cao, Liping Yuan, Yuchen Zhang, Hwee Tou Ng
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.185/
  + <details>
      <summary>Abstract</summary>
      State-of-the-art grammatical error correction (GEC) systems rely on parallel training data (ungrammatical sentences and their manually corrected counterparts), which are expensive to construct. In this paper, we employ the Break-It-Fix-It (BIFI) method to build an unsupervised GEC system. The BIFI framework generates parallel data from unlabeled text using a fixer to transform ungrammatical sentences into grammatical ones, and a critic to predict sentence grammaticality. We present an unsupervised approach to build the fixer and the critic, and an algorithm that allows them to iteratively improve each other. We evaluate our unsupervised GEC system on English and Chinese GEC. Empirical results show that our GEC system outperforms previous unsupervised GEC systems, and achieves performance comparable to supervised GEC systems without ensemble. Furthermore, when combined with labeled training data, our system achieves new state-of-the-art results on the CoNLL-2014 and NLPCC-2018 test sets.
    </details>
+ **Grammatical Error Correction via Mixed-Grained Weighted Training**
  + Authors: Jiahao Li, Quan Wang, Chiwei Zhu, Zhendong Mao, Yongdong Zhang
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2023.findings-emnlp.400/
  + <details>
      <summary>Abstract</summary>
      The task of Grammatical Error Correction (GEC) aims to automatically correct grammatical errors in natural texts. Almost all previous works treat annotated training data equally, but inherent discrepancies in data are neglected. In this paper, the inherent discrepancies are manifested in two aspects, namely, accuracy of data annotation and diversity of potential annotations. To this end, we propose MainGEC, which designs token-level and sentence-level training weights based on inherent discrepancies therein, and then conducts mixed-grained weighted training to improve the training effect for GEC. Empirical evaluation shows that whether in the Seq2Seq or Seq2Edit manner, MainGEC achieves consistent and significant performance improvements on two benchmark datasets, demonstrating the effectiveness and superiority of the mixed-grained weighted training. Further ablation experiments verify the effectiveness of designed weights for both granularities in MainGEC.
    </details>
+ **Improving Seq2Seq Grammatical Error Correction via Decoding Interventions**
  + Authors: Houquan Zhou, Yumeng Liu, Zhenghua Li, Min Zhang, Bo Zhang, Chen Li, Ji Zhang, Fei Huang
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2023.findings-emnlp.495/
  + <details>
      <summary>Abstract</summary>
      The sequence-to-sequence (Seq2Seq) approach has recently been widely used in grammatical error correction (GEC) and shows promising performance. However, the Seq2Seq GEC approach still suffers from two issues. First, a Seq2Seq GEC model can only be trained on parallel data, which, in GEC task, is often noisy and limited in quantity. Second, the decoder of a Seq2Seq GEC model lacks an explicit awareness of the correctness of the token being generated. In this paper, we propose a unified decoding intervention framework that employs an external critic to assess the appropriateness of the token to be generated incrementally, and then dynamically influence the choice of the next token. We discover and investigate two types of critics: a pre-trained left-to-right language model critic and an incremental target-side grammatical error detector critic. Through extensive experiments on English and Chinese datasets, our framework consistently outperforms strong baselines and achieves results competitive with state-of-the-art methods.
    </details>
+ **MixEdit: Revisiting Data Augmentation and Beyond for Grammatical Error Correction**
  + Authors: Jingheng Ye, Yinghui Li, Yangning Li, Hai-Tao Zheng
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2023.findings-emnlp.681/
  + <details>
      <summary>Abstract</summary>
      Data Augmentation through generating pseudo data has been proven effective in mitigating the challenge of data scarcity in the field of Grammatical Error Correction (GEC). Various augmentation strategies have been widely explored, most of which are motivated by two heuristics, i.e., increasing the distribution similarity and diversity of pseudo data. However, the underlying mechanism responsible for the effectiveness of these strategies remains poorly understood. In this paper, we aim to clarify how data augmentation improves GEC models. To this end, we introduce two interpretable and computationally efficient measures: Affinity and Diversity. Our findings indicate that an excellent GEC data augmentation strategy characterized by high Affinity and appropriate Diversity can better improve the performance of GEC models. Based on this observation, we propose MixEdit, a data augmentation approach that strategically and dynamically augments realistic data, without requiring extra monolingual corpora. To verify the correctness of our findings and the effectiveness of the proposed MixEdit, we conduct experiments on mainstream English and Chinese GEC datasets. The results show that MixEdit substantially improves GEC models and is complementary to traditional data augmentation methods. All the source codes of MixEdit are released at https://github.com/THUKElab/MixEdit.
    </details>
+ **A Low-Resource Approach to the Grammatical Error Correction of Ukrainian**
  + Authors: Frank Palma Gomez, Alla Rozovskaya, Dan Roth
  + Conference: EACL
  + Link: https://aclanthology.org/2023.unlp-1.14/
  + <details>
      <summary>Abstract</summary>
      We present our system that participated in the shared task on the grammatical error correction of Ukrainian. We have implemented two approaches that make use of large pre-trained language models and synthetic data, that have been used for error correction of English as well as low-resource languages. The first approach is based on fine-tuning a large multilingual language model (mT5) in two stages: first, on synthetic data, and then on gold data. The second approach trains a (smaller) seq2seq Transformer model pre-trained on synthetic data and fine-tuned on gold data. Our mT5-based model scored first in “GEC only” track, and a very close second in the “GEC+Fluency” track. Our two key innovations are (1) finetuning in stages, first on synthetic, and then on gold data; and (2) a high-quality corruption method based on roundtrip machine translation to complement existing noisification approaches.
    </details>
+ **ALLECS: A Lightweight Language Error Correction System**
  + Authors: Muhammad Reza Qorib, Geonsik Moon, Hwee Tou Ng
  + Conference: EACL
  + Link: https://aclanthology.org/2023.eacl-demo.32/
  + <details>
      <summary>Abstract</summary>
      In this paper, we present ALLECS, a lightweight web application to serve grammatical error correction (GEC) systems so that they can be easily used by the general public. We design ALLECS to be accessible to as many users as possible, including users who have a slow Internet connection and who use mobile phones as their main devices to connect to the Internet. ALLECS provides three state-of-the-art base GEC systems using two approaches (sequence-to-sequence generation and sequence tagging), as well as two state-of-the-art GEC system combination methods using two approaches (edit-based and text-based). ALLECS can be accessed at https://sterling8.d2.comp.nus.edu.sg/gec-demo/
    </details>
+ **Comparative Study of Models Trained on Synthetic Data for Ukrainian Grammatical Error Correction**
  + Authors: Maksym Bondarenko, Artem Yushko, Andrii Shportko, Andrii Fedorych
  + Conference: EACL
  + Link: https://aclanthology.org/2023.unlp-1.13/
  + <details>
      <summary>Abstract</summary>
      The task of Grammatical Error Correction (GEC) has been extensively studied for the English language. However, its application to low-resource languages, such as Ukrainian, remains an open challenge. In this paper, we develop sequence tagging and neural machine translation models for the Ukrainian language as well as a set of algorithmic correction rules to augment those systems. We also develop synthetic data generation techniques for the Ukrainian language to create high-quality human-like errors. Finally, we determine the best combination of synthetically generated data to augment the existing UA-GEC corpus and achieve the state-of-the-art results of 0.663 F0.5 score on the newly established UA-GEC benchmark. The code and trained models will be made publicly available on GitHub and HuggingFace.
    </details>
+ **Mitigating Exposure Bias in Grammatical Error Correction with Data Augmentation and Reweighting**
  + Authors: Hannan Cao, Wenmian Yang, Hwee Tou Ng
  + Conference: EACL
  + Link: https://aclanthology.org/2023.eacl-main.155/
  + <details>
      <summary>Abstract</summary>
      The most popular approach in grammatical error correction (GEC) is based on sequence-to-sequence (seq2seq) models. Similar to other autoregressive generation tasks, seq2seq GEC also faces the exposure bias problem, i.e., the context tokens are drawn from different distributions during training and testing, caused by the teacher forcing mechanism. In this paper, we propose a novel data manipulation approach to overcome this problem, which includes a data augmentation method during training to mimic the decoder input at inference time, and a data reweighting method to automatically balance the importance of each kind of augmented samples. Experimental results on benchmark GEC datasets show that our method achieves significant improvements compared to prior approaches.
    </details>
+ **RedPenNet for Grammatical Error Correction: Outputs to Tokens, Attentions to Spans**
  + Authors: Bohdan Didenko, Andrii Sameliuk
  + Conference: EACL
  + Link: https://aclanthology.org/2023.unlp-1.15/
  + <details>
      <summary>Abstract</summary>
      The text editing tasks, including sentence fusion, sentence splitting and rephrasing, text simplification, and Grammatical Error Correction (GEC), share a common trait of dealing with highly similar input and output sequences. This area of research lies at the intersection of two well-established fields: (i) fully autoregressive sequence-to-sequence approaches commonly used in tasks like Neural Machine Translation (NMT) and (ii) sequence tagging techniques commonly used to address tasks such as Part-of-speech tagging, Named-entity recognition (NER), and similar. In the pursuit of a balanced architecture, researchers have come up with numerous imaginative and unconventional solutions, which we’re discussing in the Related Works section. Our approach to addressing text editing tasks is called RedPenNet and is aimed at reducing architectural and parametric redundancies presented in specific Sequence-To-Edits models, preserving their semi-autoregressive advantages. Our models achieve F0.5 scores of 77.60 on the BEA-2019 (test), which can be considered as state-of-the-art the only exception for system combination (Qorib et al., 2022) and 67.71 on the UAGEC+Fluency (test) benchmarks. This research is being conducted in the context of the UNLP 2023 workshop, where it will be presented as a paper for the Shared Task in Grammatical Error Correction (GEC) for Ukrainian. This study aims to apply the RedPenNet approach to address the GEC problem in the Ukrainian language.
    </details>
+ **The UNLP 2023 Shared Task on Grammatical Error Correction for Ukrainian**
  + Authors: Oleksiy Syvokon, Mariana Romanyshyn
  + Conference: EACL
  + Link: https://aclanthology.org/2023.unlp-1.16/
  + <details>
      <summary>Abstract</summary>
      This paper presents the results of the UNLP 2023 shared task, the first Shared Task on Grammatical Error Correction for the Ukrainian language. The task included two tracks: GEC-only and GEC+Fluency. The dataset and evaluation scripts were provided to the participants, and the final results were evaluated on a hidden test set. Six teams submitted their solutions before the deadline, and four teams submitted papers that were accepted to appear in the UNLP workshop proceedings and are referred to in this report. The CodaLab leaderboard is left open for further submissions.
    </details>
+ **Towards Automatic Grammatical Error Type Classification for Turkish**
  + Authors: Harun Uz, Gülşen Eryiğit
  + Conference: EACL
  + Link: https://aclanthology.org/2023.eacl-srw.14/
  + <details>
      <summary>Abstract</summary>
      Automatic error type classification is an important process in both learner corpora creation and evaluation of large-scale grammatical error correction systems. Rule-based classifier approaches such as ERRANT have been widely used to classify edits between correct-erroneous sentence pairs into predefined error categories. However, the used error categories are far from being universal yielding many language specific variants of ERRANT.In this paper, we discuss the applicability of the previously introduced grammatical error types to an agglutinative language, Turkish. We suggest changes on current error categories and discuss a hierarchical structure to better suit the inflectional and derivational properties of this morphologically highly rich language. We also introduce ERRANT-TR, the first automatic error type classification toolkit for Turkish. ERRANT-TR currently uses a rule-based error type classification pipeline which relies on word level morphological information. Due to unavailability of learner corpora in Turkish, the proposed system is evaluated on a small set of 106 annotated sentences and its performance is measured as 77.04% F0.5 score. The next step is to use ERRANT-TR for the development of a Turkish learner corpus.
    </details>
+ **UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language**
  + Authors: Oleksiy Syvokon, Olena Nahorna, Pavlo Kuchmiichuk, Nastasiia Osidach
  + Conference: EACL
  + Link: https://aclanthology.org/2023.unlp-1.12/
  + <details>
      <summary>Abstract</summary>
      We present a corpus professionally annotated for grammatical error correction (GEC) and fluency edits in the Ukrainian language. We have built two versions of the corpus – GEC+Fluency and GEC-only – to differentiate the corpus application. To the best of our knowledge, this is the first GEC corpus for the Ukrainian language. We collected texts with errors (33,735 sentences) from a diverse pool of contributors, including both native and non-native speakers. The data cover a wide variety of writing domains, from text chats and essays to formal writing. Professional proofreaders corrected and annotated the corpus for errors relating to fluency, grammar, punctuation, and spelling. This corpus can be used for developing and evaluating GEC systems in Ukrainian. More generally, it can be used for researching multilingual and low-resource NLP, morphologically rich languages, document-level GEC, and fluency correction. The corpus is publicly available at https://github.com/grammarly/ua-gec
    </details>
+ **An Extended Sequence Tagging Vocabulary for Grammatical Error Correction**
  + Authors: Stuart Mesham, Christopher Bryant, Marek Rei, Zheng Yuan
  + Conference: EACL Findings
  + Link: https://aclanthology.org/2023.findings-eacl.119/
  + <details>
      <summary>Abstract</summary>
      We extend a current sequence-tagging approach to Grammatical Error Correction (GEC) by introducing specialised tags for spelling correction and morphological inflection using the SymSpell and LemmInflect algorithms. Our approach improves generalisation: the proposed new tagset allows a smaller number of tags to correct a larger range of errors. Our results show a performance improvement both overall and in the targeted error categories. We further show that ensembles trained with our new tagset outperform those trained with the baseline tagset on the public BEA benchmark.
    </details>
+ **Grammatical Error Correction through Round-Trip Machine Translation**
  + Authors: Yova Kementchedjhieva, Anders Søgaard
  + Conference: EACL Findings
  + Link: https://aclanthology.org/2023.findings-eacl.165/
  + <details>
      <summary>Abstract</summary>
      Machine translation (MT) operates on the premise of an interlingua which abstracts away from surface form while preserving meaning. A decade ago the idea of using round-trip MT to guide grammatical error correction was proposed as a way to abstract away from potential errors in surface forms (Madnani et al., 2012). At the time, it did not pan out due to the low quality of MT systems of the day. Today much stronger MT systems are available so we re-evaluate this idea across five languages and models of various sizes. We find that for extra large models input augmentation through round-trip MT has little to no effect. For more ‘workable’ model sizes, however, it yields consistent improvements, sometimes bringing the performance of a base or large model up to that of a large or xl model, respectively. The round-trip translation comes at a computational cost though, so one would have to determine whether to opt for a larger model or for input augmentation on a case-by-case basis.
    </details>
+ **Balarila: Deep Learning for Semantic Grammar Error Correction in Low-Resource Settings**
  + Authors: Andre Dominic H. Ponce, Joshue Salvador A. Jadie, Paolo Edni Andryn Espiritu, Charibeth Cheng
  + Conference: AACL
  + Link: https://aclanthology.org/2023.sealp-1.3/

+ **CCL23-Eval 任务7赛道一系统报告:基于序列到序列模型的自动化文本纠错系统(System Report for CCL23-Eval Task 7 Track 1: Automated text error correction pipeline based on sequence-to-sequence models)**
  + Authors: Shixuan Liu (刘世萱), Xinzhang Liu (刘欣璋), Yuyao Huang (黄钰瑶), Chao Wang (王超), Yongshuang Song (宋双永)
  + Conference: CCL
  + Link: https://aclanthology.org/2023.ccl-3.24/
  + <details>
      <summary>Abstract</summary>
      “本文介绍了本队伍在CCL-2023汉语学习者文本纠错评测大赛赛道一中提交的参赛系统。近年来,大规模的中文预训练模型在各种任务上表现出色,而不同的预训练模型在特定任务上也各有优势。然而,由于汉语学习者文本纠错任务存在语法错误复杂和纠错语料稀缺等特点,因此采用基于序列标记的预训练文本纠错模型来解决问题是自然的选择。我们的团队采用了序列到序列的纠错模型,并采取了两阶段训练策略,设计了一套基于序列到序列文本纠错的pipeline。首先,我们对训练集数据进行了清洗处理;在第一阶段训练中,我们在训练集上使用数据增强技术;在第二阶段,我们利用验证集进行微调,并最终采用多个模型投票集成的方式完成后处理。在实际的系统测评中,我们提交的结果在封闭任务排行榜上超出baseline模型17.01分(40.59->57.6)。”
    </details>
+ **CCL23-Eval任务7赛道一系统报告:Suda &Alibaba 文本纠错系统(CCL23-Eval Task 7 Track 1 System Report: Suda &Alibaba Team Text Error Correction System)**
  + Authors: Haochen Jiang (蒋浩辰), Yumeng Liu (刘雨萌), Houquan Zhou (周厚全), Ziheng Qiao (乔子恒), Bo Zhang (波章,), Chen Li (李辰), Zhenghua Li (李正华), Min Zhang (张民)
  + Conference: CCL
  + Link: https://aclanthology.org/2023.ccl-3.25/
  + <details>
      <summary>Abstract</summary>
      “本报告描述 Suda &Alibaba 纠错团队在 CCL2023 汉语学习者文本纠错评测任务的赛道一:多维度汉语学习者文本纠错(Multidimensional Chinese Learner Text Correc-tion)中提交的参赛系统。在模型方面,本队伍使用了序列到序列和序列到编辑两种纠错模型。在数据方面,本队伍分别使用基于混淆集构造的伪数据、Lang-8 真实数据以及 YACLC 开发集进行三阶段训练;在开放任务上还额外使用HSK、CGED等数据进行训练。本队伍还使用了一系列有效的性能提升技术,包括了基于规则的数据增强,数据清洗,后处理以及模型集成等 .除此之外,本队伍还在如何使用GPT3.5、GPT4等大模型来辅助中文文本纠错上进行了一些探索,提出了一种可以有效避免大模型过纠问题的方法,并尝试了多种 Prompt。在封闭和开放两个任务上,本队伍在最小改动、流利提升和平均 F0.5 得分上均位列第一。”
    </details>
+ **Effectiveness of ChatGPT in Korean Grammatical Error Correction**
  + Authors: Junghwan Maeng, Jinghang Gu, Sun-A Kim
  + Conference: PACLIC
  + Link: https://aclanthology.org/2023.paclic-1.46/

+ **GECTurk: Grammatical Error Correction and Detection Dataset for Turkish**
  + Authors: Atakan Kara, Farrin Marouf Sofian, Andrew Bond, Gözde Şahin
  + Conference: AACL Findings
  + Link: https://aclanthology.org/2023.findings-ijcnlp.26/

+ **Grammatical Error Correction: A Survey of the State of the Art**
  + Authors: Christopher Bryant, Zheng Yuan, Muhammad Reza Qorib, Hannan Cao, Hwee Tou Ng, Ted Briscoe
  + Conference: CL
  + Link: https://aclanthology.org/2023.cl-3.4/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is the task of automatically detecting and correcting errors in text. The task not only includes the correction of grammatical errors, such as missing prepositions and mismatched subject–verb agreement, but also orthographic and semantic errors, such as misspellings and word choice errors, respectively. The field has seen significant progress in the last decade, motivated in part by a series of five shared tasks, which drove the development of rule-based methods, statistical classifiers, statistical machine translation, and finally neural machine translation systems, which represent the current dominant state of the art. In this survey paper, we condense the field into a single article and first outline some of the linguistic challenges of the task, introduce the most popular datasets that are available to researchers (for both English and other languages), and summarize the various methods and techniques that have been developed with a particular focus on artificial error generation. We next describe the many different approaches to evaluation as well as concerns surrounding metric reliability, especially in relation to subjective human judgments, before concluding with an overview of recent progress and suggestions for future work and remaining challenges. We hope that this survey will serve as a comprehensive resource for researchers who are new to the field or who want to be kept apprised of recent developments.
    </details>
+ **Minimum Bayes’ Risk Decoding for System Combination of Grammatical Error Correction Systems**
  + Authors: Vyas Raina, Mark Gales
  + Conference: AACL
  + Link: https://aclanthology.org/2023.ijcnlp-short.12/

+ **System Report for CCL23-Eval Task 7: THU KELab (sz) - Exploring Data Augmentation and Denoising for Chinese Grammatical Error Correction**
  + Authors: Jingheng Ye, Yinghui Li, Haitao Zheng
  + Conference: CCL
  + Link: https://aclanthology.org/2023.ccl-3.29/
  + <details>
      <summary>Abstract</summary>
      “This paper explains our GEC system submitted by THU KELab (sz) in the CCL2023-Eval Task7 CLTC (Chinese Learner Text Correction) Track 1: Multidimensional Chinese Learner TextCorrection. Recent studies have demonstrate GEC performance can be improved by increasingthe amount of training data. However, high-quality public GEC data is much less abundant. To address this issue, we propose two data-driven techniques, data augmentation and data de-noising, to improve the GEC performance. Data augmentation creates pseudo data to enhancegeneralization, while data denoising removes noise from the realistic training data. The resultson the official evaluation dataset YACLC demonstrate the effectiveness of our approach. Finally,our GEC system ranked second in both close and open tasks. All of our datasets and codes areavailabel at https://github.com/THUKElab/CCL2023-CLTC-THU_KELab.”
    </details>
## 2022
+ **Adjusting the Precision-Recall Trade-Off with Align-and-Predict Decoding for Grammatical Error Correction**
  + Authors: Xin Sun, Houfeng Wang
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-short.77/
  + <details>
      <summary>Abstract</summary>
      Modern writing assistance applications are always equipped with a Grammatical Error Correction (GEC) model to correct errors in user-entered sentences. Different scenarios have varying requirements for correction behavior, e.g., performing more precise corrections (high precision) or providing more candidates for users (high recall). However, previous works adjust such trade-off only for sequence labeling approaches. In this paper, we propose a simple yet effective counterpart – Align-and-Predict Decoding (APD) for the most popular sequence-to-sequence models to offer more flexibility for the precision-recall trade-off. During inference, APD aligns the already generated sequence with input and adjusts scores of the following tokens. Experiments in both English and Chinese GEC benchmarks show that our approach not only adapts a single model to precision-oriented and recall-oriented inference, but also maximizes its potential to achieve state-of-the-art results. Our code is available at https://github.com/AutoTemp/Align-and-Predict.
    </details>
+ **Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction**
  + Authors: Maksym Tarnavskyi, Artem Chernodub, Kostiantyn Omelianchuk
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-long.266/
  + <details>
      <summary>Abstract</summary>
      In this paper, we investigate improvements to the GEC sequence tagging architecture with a focus on ensembling of recent cutting-edge Transformer-based encoders in Large configurations. We encourage ensembling models by majority votes on span-level edits because this approach is tolerant to the model architecture and vocabulary size. Our best ensemble achieves a new SOTA result with an F0.5 score of 76.05 on BEA-2019 (test), even without pre-training on synthetic datasets. In addition, we perform knowledge distillation with a trained ensemble to generate new synthetic training datasets, “Troy-Blogs” and “Troy-1BW”. Our best single sequence tagging model that is pretrained on the generated Troy- datasets in combination with the publicly available synthetic PIE dataset achieves a near-SOTA result with an F0.5 score of 73.21 on BEA-2019 (test). The code, datasets, and trained models are publicly available.
    </details>
+ **Interpretability for Language Learners Using Example-Based Grammatical Error Correction**
  + Authors: Masahiro Kaneko, Sho Takase, Ayana Niwa, Naoaki Okazaki
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-long.496/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) should not focus only on high accuracy of corrections but also on interpretability for language learning. However, existing neural-based GEC models mainly aim at improving accuracy, and their interpretability has not been explored.A promising approach for improving interpretability is an example-based method, which uses similar retrieved examples to generate corrections. In addition, examples are beneficial in language learning, helping learners understand the basis of grammatically incorrect/correct texts and improve their confidence in writing. Therefore, we hypothesize that incorporating an example-based method into GEC can improve interpretability as well as support language learners. In this study, we introduce an Example-Based GEC (EB-GEC) that presents examples to language learners as a basis for a correction result. The examples consist of pairs of correct and incorrect sentences similar to a given input and its predicted correction. Experiments demonstrate that the examples presented by EB-GEC help language learners decide to accept or refuse suggestions from the GEC output. Furthermore, the experiments also show that retrieved examples improve the accuracy of corrections.
    </details>
+ **ODE Transformer: An Ordinary Differential Equation-Inspired Model for Sequence Generation**
  + Authors: Bei Li, Quan Du, Tao Zhou, Yi Jing, Shuhan Zhou, Xin Zeng, Tong Xiao, JingBo Zhu, Xuebo Liu, Min Zhang
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-long.571/
  + <details>
      <summary>Abstract</summary>
      Residual networks are an Euler discretization of solutions to Ordinary Differential Equations (ODE). This paper explores a deeper relationship between Transformer and numerical ODE methods. We first show that a residual block of layers in Transformer can be described as a higher-order solution to ODE. Inspired by this, we design a new architecture, ODE Transformer, which is analogous to the Runge-Kutta method that is well motivated in ODE. As a natural extension to Transformer, ODE Transformer is easy to implement and efficient to use. Experimental results on the large-scale machine translation, abstractive summarization, and grammar error correction tasks demonstrate the high genericity of ODE Transformer. It can gain large improvements in model performance over strong baselines (e.g., 30.77 and 44.11 BLEU scores on the WMT’14 English-German and English-French benchmarks) at a slight cost in inference efficiency.
    </details>
+ **Uncertainty Determines the Adequacy of the Mode and the Tractability of Decoding in Sequence-to-Sequence Models**
  + Authors: Felix Stahlberg, Ilia Kulikov, Shankar Kumar
  + Conference: ACL
  + Link: https://aclanthology.org/2022.acl-long.591/
  + <details>
      <summary>Abstract</summary>
      In many natural language processing (NLP) tasks the same input (e.g. source sentence) can have multiple possible outputs (e.g. translations). To analyze how this ambiguity (also known as intrinsic uncertainty) shapes the distribution learned by neural sequence models we measure sentence-level uncertainty by computing the degree of overlap between references in multi-reference test sets from two different NLP tasks: machine translation (MT) and grammatical error correction (GEC). At both the sentence- and the task-level, intrinsic uncertainty has major implications for various aspects of search such as the inductive biases in beam search and the complexity of exact search. In particular, we show that well-known pathologies such as a high number of beam search errors, the inadequacy of the mode, and the drop in system performance with large beam sizes apply to tasks with high level of ambiguity such as MT but not to less uncertain tasks such as GEC. Furthermore, we propose a novel exact n-best search algorithm for neural sequence models, and show that intrinsic uncertainty affects model uncertainty as the model tends to overly spread out the probability mass for uncertain tasks and sentences.
    </details>
+ **Type-Driven Multi-Turn Corrections for Grammatical Error Correction**
  + Authors: Shaopeng Lai, Qingyu Zhou, Jiali Zeng, Zhongli Li, Chao Li, Yunbo Cao, Jinsong Su
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2022.findings-acl.254/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) aims to automatically detect and correct grammatical errors. In this aspect, dominant models are trained by one-iteration learning while performing multiple iterations of corrections during inference. Previous studies mainly focus on the data augmentation approach to combat the exposure bias, which suffers from two drawbacks. First, they simply mix additionally-constructed training instances and original ones to train models, which fails to help models be explicitly aware of the procedure of gradual corrections. Second, they ignore the interdependence between different types of corrections. In this paper, we propose a Type-Driven Multi-Turn Corrections approach for GEC. Using this approach, from each training instance, we additionally construct multiple training instances, each of which involves the correction of a specific type of errors. Then, we use these additionally-constructed training instances and the original one to train the model in turn. Experimental results and in-depth analysis show that our approach significantly benefits the model training. Particularly, our enhanced model achieves state-of-the-art single-model performance on English GEC benchmarks. We release our code at Github.
    </details>
+ **“Is Whole Word Masking Always Better for Chinese BERT?”: Probing on Chinese Grammatical Error Correction**
  + Authors: Yong Dai, Linyang Li, Cong Zhou, Zhangyin Feng, Enbo Zhao, Xipeng Qiu, Piji Li, Duyu Tang
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2022.findings-acl.1/
  + <details>
      <summary>Abstract</summary>
      Whole word masking (WWM), which masks all subwords corresponding to a word at once, makes a better English BERT model. For the Chinese language, however, there is no subword because each token is an atomic character. The meaning of a word in Chinese is different in that a word is a compositional unit consisting of multiple characters. Such difference motivates us to investigate whether WWM leads to better context understanding ability for Chinese BERT. To achieve this, we introduce two probing tasks related to grammatical error correction and ask pretrained models to revise or insert tokens in a masked language modeling manner. We construct a dataset including labels for 19,075 tokens in 10,448 sentences. We train three Chinese BERT models with standard character-level masking (CLM), WWM, and a combination of CLM and WWM, respectively. Our major findings are as follows: First, when one character needs to be inserted or replaced, the model trained with CLM performs the best. Second, when more than one character needs to be handled, WWM is the key to better performance. Finally, when being fine-tuned on sentence-level downstream tasks, models trained with different masking strategies perform comparably.
    </details>
+ **Improved grammatical error correction by ranking elementary edits**
  + Authors: Alexey Sorokin
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.785/
  + <details>
      <summary>Abstract</summary>
      We offer a two-stage reranking method for grammatical error correction: the first model serves as edit generator, while the second classifies the proposed edits as correct or false. We show how to use both encoder-decoder and sequence labeling models for the first step of our pipeline. We achieve state-of-the-art quality on BEA 2019 English dataset even using weak BERT-GEC edit generator. Combining our roberta-base scorer with state-of-the-art GECToR edit generator, we surpass GECToR by 2-3%. With a larger model we establish a new SOTA on BEA development and test sets. Our model also sets a new SOTA on Russian, despite using smaller models and less data than the previous approaches.
    </details>
+ **Improving Iterative Text Revision by Learning Where to Edit from Other Revision Tasks**
  + Authors: Zae Myung Kim, Wanyu Du, Vipul Raheja, Dhruv Kumar, Dongyeop Kang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.678/
  + <details>
      <summary>Abstract</summary>
      Iterative text revision improves text quality by fixing grammatical errors, rephrasing for better readability or contextual appropriateness, or reorganizing sentence structures throughout a document. Most recent research has focused on understanding and classifying different types of edits in the iterative revision process from human-written text instead of building accurate and robust systems for iterative text revision. In this work, we aim to build an end-to-end text revision system that can iteratively generate helpful edits by explicitly detecting editable spans (where-to-edit) with their corresponding edit intents and then instructing a revision model to revise the detected edit spans. Leveraging datasets from other related text editing NLP tasks, combined with the specification of editable spans, leads our system to more accurately model the process of iterative text refinement, as evidenced by empirical results and human evaluations. Our system significantly outperforms previous baselines on our text revision tasks and other standard text revision tasks, including grammatical error correction, text simplification, sentence fusion, and style transfer. Through extensive qualitative and quantitative analysis, we make vital connections between edit intentions and writing quality, and better computational modeling of iterative text revisions.
    </details>
+ **Mask the Correct Tokens: An Embarrassingly Simple Approach for Error Correction**
  + Authors: Kai Shen, Yichong Leng, Xu Tan, Siliang Tang, Yuan Zhang, Wenjie Liu, Edward Lin
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.708/
  + <details>
      <summary>Abstract</summary>
      Text error correction aims to correct the errors in text sequences such as those typed by humans or generated by speech recognition models. Previous error correction methods usually take the source (incorrect) sentence as encoder input and generate the target (correct) sentence through the decoder. Since the error rate of the incorrect sentence is usually low (e.g., 10%), the correction model can only learn to correct on limited error tokens but trivially copy on most tokens (correct tokens), which harms the effective training of error correction. In this paper, we argue that the correct tokens should be better utilized to facilitate effective training and then propose a simple yet effective masking strategy to achieve this goal. Specifically, we randomly mask out a part of the correct tokens in the source sentence and let the model learn to not only correct the original error tokens but also predict the masked tokens based on their context information. Our method enjoys several advantages: 1) it alleviates trivial copy; 2) it leverages effective training signals from correct tokens; 3) it is a plug-and-play module and can be applied to different models and tasks. Experiments on spelling error correction and speech recognition error correction on Mandarin datasets and grammar error correction on English datasets with both autoregressive and non-autoregressive generation models show that our method improves the correctionaccuracy consistently.
    </details>
+ **Revisiting Grammatical Error Correction Evaluation and Beyond**
  + Authors: Peiyuan Gong, Xuebo Liu, Heyan Huang, Min Zhang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.463/
  + <details>
      <summary>Abstract</summary>
      Pretraining-based (PT-based) automatic evaluation metrics (e.g., BERTScore and BARTScore) have been widely used in several sentence generation tasks (e.g., machine translation and text summarization) due to their better correlation with human judgments over traditional overlap-based methods. Although PT-based methods have become the de facto standard for training grammatical error correction (GEC) systems, GEC evaluation still does not benefit from pretrained knowledge. This paper takes the first step towards understanding and improving GEC evaluation with pretraining. We first find that arbitrarily applying PT-based metrics to GEC evaluation brings unsatisfactory correlation results because of the excessive attention to inessential systems outputs (e.g., unchanged parts). To alleviate the limitation, we propose a novel GEC evaluation metric to achieve the best of both worlds, namely PT-M2 which only uses PT-based metrics to score those corrected parts. Experimental results on the CoNLL14 evaluation task show that PT-M2 significantly outperforms existing methods, achieving a new state-of-the-art result of 0.949 Pearson correlation. Further analysis reveals that PT-M2 is robust to evaluate competitive GEC systems. Source code and scripts are freely available at https://github.com/pygongnlp/PT-M2.
    </details>
+ **SynGEC: Syntax-Enhanced Grammatical Error Correction with a Tailored GEC-Oriented Parser**
  + Authors: Yue Zhang, Bo Zhang, Zhenghua Li, Zuyi Bao, Chen Li, Min Zhang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2022.emnlp-main.162/
  + <details>
      <summary>Abstract</summary>
      This work proposes a syntax-enhanced grammatical error correction (GEC) approach named SynGEC that effectively incorporates dependency syntactic information into the encoder part of GEC models. The key challenge for this idea is that off-the-shelf parsers are unreliable when processing ungrammatical sentences. To confront this challenge, we propose to build a tailored GEC-oriented parser (GOPar) using parallel GEC training data as a pivot. First, we design an extended syntax representation scheme that allows us to represent both grammatical errors and syntax in a unified tree structure. Then, we obtain parse trees of the source incorrect sentences by projecting trees of the target correct sentences. Finally, we train GOPar with such projected trees. For GEC, we employ the graph convolution network to encode source-side syntactic information produced by GOPar, and fuse them with the outputs of the Transformer encoder. Experiments on mainstream English and Chinese GEC datasets show that our proposed SynGEC approach consistently and substantially outperforms strong baselines and achieves competitive performance. Our code and data are all publicly available at https://github.com/HillZhang1999/SynGEC.
    </details>
+ **EdiT5: Semi-Autoregressive Text Editing with T5 Warm-Start**
  + Authors: Jonathan Mallinson, Jakub Adamek, Eric Malmi, Aliaksei Severyn
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.156/
  + <details>
      <summary>Abstract</summary>
      We present EdiT5 - a novel semi-autoregressive text-editing approach designed to combine the strengths of non-autoregressive text-editing and autoregressive decoding. EdiT5 is faster at inference times than conventional sequence-to-sequence (seq2seq) models, while being capable of modeling flexible input-output transformations. This is achieved by decomposing the generation process into three sub-tasks: (1) tagging to decide on the subset of input tokens to be preserved in the output, (2) re-ordering to define their order in the output text, and (3) insertion to infill the missing tokens that are not present in the input. The tagging and re-ordering steps, which are responsible for generating the largest portion of the output, are non-autoregressive, while the insertion uses an autoregressive decoder. Depending on the task, EdiT5 requires significantly fewer autoregressive steps demonstrating speedups of up to 25x when compared to classic seq2seq models. Quality-wise, EdiT5 is initialized with a pre-trained T5 checkpoint yielding comparable performance to T5 in high-resource settings and clearly outperforms it on low-resource settings when evaluated on three NLG tasks: Sentence Fusion, Grammatical Error Correction, and Decontextualization.
    </details>
+ **FCGEC: Fine-Grained Corpus for Chinese Grammatical Error Correction**
  + Authors: Lvxiaowei Xu, Jianwang Wu, Jiawei Peng, Jiayu Fu, Ming Cai
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.137/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) has been broadly applied in automatic correction and proofreading system recently. However, it is still immature in Chinese GEC due to limited high-quality data from native speakers in terms of category and scale. In this paper, we present FCGEC, a fine-grained corpus to detect, identify and correct the grammatical errors. FCGEC is a human-annotated corpus with multiple references, consisting of 41,340 sentences collected mainly from multi-choice questions in public school Chinese examinations. Furthermore, we propose a Switch-Tagger-Generator (STG) baseline model to correct the grammatical errors in low-resource settings. Compared to other GEC benchmark models, experimental results illustrate that STG outperforms them on our FCGEC. However, there exists a significant gap between benchmark models and humans that encourages future models to bridge it.
    </details>
+ **From Spelling to Grammar: A New Framework for Chinese Grammatical Error Correction**
  + Authors: Xiuyu Wu, Yunfang Wu
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.63/
  + <details>
      <summary>Abstract</summary>
      Chinese Grammatical Error Correction (CGEC) aims to generate a correct sentence from an erroneous sequence, where different kinds of errors are mixed. This paper divides the CGEC task into two steps, namely spelling error correction and grammatical error correction. We firstly propose a novel zero-shot approach for spelling error correction, which is simple but effective, obtaining a high precision to avoid error accumulation of the pipeline structure. To handle grammatical error correction, we design part-of-speech (POS) features and semantic class features to enhance the neural network model, and propose an auxiliary task to predict the POS sequence of the target sentence. Our proposed framework achieves a 42.11 F-0.5 score on CGEC dataset without using any synthetic data or data augmentation methods, which outperforms the previous state-of-the-art by a wide margin of 1.30 points. Moreover, our model produces meaningful POS representations that capture different POS words and convey reasonable POS transition rules.
    </details>
+ **Linguistic Rules-Based Corpus Generation for Native Chinese Grammatical Error Correction**
  + Authors: Shirong Ma, Yinghui Li, Rongyi Sun, Qingyu Zhou, Shulin Huang, Ding Zhang, Li Yangning, Ruiyang Liu, Zhongli Li, Yunbo Cao, Haitao Zheng, Ying Shen
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.40/
  + <details>
      <summary>Abstract</summary>
      Chinese Grammatical Error Correction (CGEC) is both a challenging NLP task and a common application in human daily life. Recently, many data-driven approaches are proposed for the development of CGEC research. However, there are two major limitations in the CGEC field: First, the lack of high-quality annotated training corpora prevents the performance of existing CGEC models from being significantly improved. Second, the grammatical errors in widely used test sets are not made by native Chinese speakers, resulting in a significant gap between the CGEC models and the real application. In this paper, we propose a linguistic rules-based approach to construct large-scale CGEC training corpora with automatically generated grammatical errors. Additionally, we present a challenging CGEC benchmark derived entirely from errors made by native Chinese speakers in real-world scenarios. Extensive experiments and detailed analyses not only demonstrate that the training data constructed by our method effectively improves the performance of CGEC models, but also reflect that our benchmark is an excellent resource for further development of the CGEC field.
    </details>
+ **Text Editing as Imitation Game**
  + Authors: Ning Shi, Bin Tang, Bo Yuan, Longtao Huang, Yewen Pu, Jie Fu, Zhouhan Lin
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2022.findings-emnlp.114/
  + <details>
      <summary>Abstract</summary>
      Text editing, such as grammatical error correction, arises naturally from imperfect textual data. Recent works frame text editing as a multi-round sequence tagging task, where operations – such as insertion and substitution – are represented as a sequence of tags. While achieving good results, this encoding is limited in flexibility as all actions are bound to token-level tags. In this work, we reformulate text editing as an imitation game using behavioral cloning. Specifically, we convert conventional sequence-to-sequence data into state-to-action demonstrations, where the action space can be as flexible as needed. Instead of generating the actions one at a time, we introduce a dual decoders structure to parallel the decoding while retaining the dependencies between action tokens, coupled with trajectory augmentation to alleviate the distribution shift that imitation learning often suffers. In experiments on a suite of Arithmetic Equation benchmarks, our model consistently outperforms the autoregressive baselines in terms of performance, efficiency, and robustness. We hope our findings will shed light on future studies in reinforcement learning applying sequence-level action generation to natural language processing.
    </details>
+ **ErAConD: Error Annotated Conversational Dialog Dataset for Grammatical Error Correction**
  + Authors: Xun Yuan, Derek Pham, Sam Davidson, Zhou Yu
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.5/
  + <details>
      <summary>Abstract</summary>
      Currently available grammatical error correction (GEC) datasets are compiled using essays or other long-form text written by language learners, limiting the applicability of these datasets to other domains such as informal writing and conversational dialog. In this paper, we present a novel GEC dataset consisting of parallel original and corrected utterances drawn from open-domain chatbot conversations; this dataset is, to our knowledge, the first GEC dataset targeted to a human-machine conversational setting. We also present a detailed annotation scheme which ranks errors by perceived impact on comprehension, making our dataset more representative of real-world language learning applications. To demonstrate the utility of the dataset, we use our annotated data to fine-tune a state-of-the-art GEC model. Experimental results show the effectiveness of our data in improving GEC model performance in a conversational scenario.
    </details>
+ **Frustratingly Easy System Combination for Grammatical Error Correction**
  + Authors: Muhammad Reza Qorib, Seung-Hoon Na, Hwee Tou Ng
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.143/
  + <details>
      <summary>Abstract</summary>
      In this paper, we formulate system combination for grammatical error correction (GEC) as a simple machine learning task: binary classification. We demonstrate that with the right problem formulation, a simple logistic regression algorithm can be highly effective for combining GEC models. Our method successfully increases the F0.5 score from the highest base GEC system by 4.2 points on the CoNLL-2014 test set and 7.2 points on the BEA-2019 test set. Furthermore, our method outperforms the state of the art by 4.0 points on the BEA-2019 test set, 1.2 points on the CoNLL-2014 test set with original annotation, and 3.4 points on the CoNLL-2014 test set with alternative annotation. We also show that our system combination generates better corrections with higher F0.5 scores than the conventional ensemble.
    </details>
+ **MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction**
  + Authors: Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, Min Zhang
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.227/
  + <details>
      <summary>Abstract</summary>
      This paper presents MuCGEC, a multi-reference multi-source evaluation dataset for Chinese Grammatical Error Correction (CGEC), consisting of 7,063 sentences collected from three Chinese-as-a-Second-Language (CSL) learner sources. Each sentence is corrected by three annotators, and their corrections are carefully reviewed by a senior annotator, resulting in 2.3 references per sentence. We conduct experiments with two mainstream CGEC models, i.e., the sequence-to-sequence model and the sequence-to-edit model, both enhanced with large pretrained language models, achieving competitive benchmark performance on previous and our datasets. We also discuss CGEC evaluation methodologies, including the effect of multiple references and using a char-based metric. Our annotation guidelines, data, and code are available at https://github.com/HillZhang1999/MuCGEC.
    </details>
+ **On Assessing and Developing Spoken ’Grammatical Error Correction’ Systems**
  + Authors: Yiting Lu, Stefano Bannò, Mark Gales
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.bea-1.9/
  + <details>
      <summary>Abstract</summary>
      Spoken ‘grammatical error correction’ (SGEC) is an important process to provide feedback for second language learning. Due to a lack of end-to-end training data, SGEC is often implemented as a cascaded, modular system, consisting of speech recognition, disfluency removal, and grammatical error correction (GEC). This cascaded structure enables efficient use of training data for each module. It is, however, difficult to compare and evaluate the performance of individual modules as preceeding modules may introduce errors. For example the GEC module input depends on the output of non-native speech recognition and disfluency detection, both challenging tasks for learner data. This paper focuses on the assessment and development of SGEC systems. We first discuss metrics for evaluating SGEC, both individual modules and the overall system. The system-level metrics enable tuning for optimal system performance. A known issue in cascaded systems is error propagation between modules. To mitigate this problem semi-supervised approaches and self-distillation are investigated. Lastly, when SGEC system gets deployed it is important to give accurate feedback to users. Thus, we apply filtering to remove edits with low-confidence, aiming to improve overall feedback precision. The performance metrics are examined on a Linguaskill multi-level data set, which includes the original non-native speech, manual transcriptions and reference grammatical error corrections, to enable system analysis and development.
    </details>
+ **Text Generation with Text-Editing Models**
  + Authors: Eric Malmi, Yue Dong, Jonathan Mallinson, Aleksandr Chuklin, Jakub Adamek, Daniil Mirylenka, Felix Stahlberg, Sebastian Krause, Shankar Kumar, Aliaksei Severyn
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-tutorials.1/
  + <details>
      <summary>Abstract</summary>
      Text-editing models have recently become a prominent alternative to seq2seq models for monolingual text-generation tasks such as grammatical error correction, text simplification, and style transfer. These tasks share a common trait – they exhibit a large amount of textual overlap between the source and target texts. Text-editing models take advantage of this observation and learn to generate the output by predicting edit operations applied to the source sequence. In contrast, seq2seq models generate outputs word-by-word from scratch thus making them slow at inference time. Text-editing models provide several benefits over seq2seq models including faster inference speed, higher sample efficiency, and better control and interpretability of the outputs. This tutorial provides a comprehensive overview of the text-edit based models and current state-of-the-art approaches analyzing their pros and cons. We discuss challenges related to deployment and how these models help to mitigate hallucination and bias, both pressing challenges in the field of text generation.
    </details>
+ **CCTC: A Cross-Sentence Chinese Text Correction Dataset for Native Speakers**
  + Authors: Baoxin Wang, Xingyi Duan, Dayong Wu, Wanxiang Che, Zhigang Chen, Guoping Hu
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.294/
  + <details>
      <summary>Abstract</summary>
      The Chinese text correction (CTC) focuses on detecting and correcting Chinese spelling errors and grammatical errors. Most existing datasets of Chinese spelling check (CSC) and Chinese grammatical error correction (GEC) are focused on a single sentence written by Chinese-as-a-second-language (CSL) learners. We find that errors caused by native speakers differ significantly from those produced by non-native speakers. These differences make it inappropriate to use the existing test sets directly to evaluate text correction systems for native speakers. Some errors also require the cross-sentence information to be identified and corrected. In this paper, we propose a cross-sentence Chinese text correction dataset for native speakers. Concretely, we manually annotated 1,500 texts written by native speakers. The dataset consists of 30,811 sentences and more than 1,000,000 Chinese characters. It contains four types of errors: spelling errors, redundant words, missing words, and word ordering errors. We also test some state-of-the-art models on the dataset. The experimental results show that even the model with the best performance is 20 points lower than humans, which indicates that there is still much room for improvement. We hope that the new dataset can fill the gap in cross-sentence text correction for native Chinese speakers.
    </details>
+ **Grammatical Error Correction: Are We There Yet?**
  + Authors: Muhammad Reza Qorib, Hwee Tou Ng
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.246/
  + <details>
      <summary>Abstract</summary>
      There has been much recent progress in natural language processing, and grammatical error correction (GEC) is no exception. We found that state-of-the-art GEC systems (T5 and GECToR) outperform humans by a wide margin on the CoNLL-2014 test set, a benchmark GEC test corpus, as measured by the standard F0.5 evaluation metric. However, a careful examination of their outputs reveals that there are still classes of errors that they fail to correct. This suggests that creating new test data that more accurately measure the true performance of GEC systems constitutes important future work.
    </details>
+ **IMPARA: Impact-Based Metric for GEC Using Parallel Data**
  + Authors: Koki Maeda, Masahiro Kaneko, Naoaki Okazaki
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.316/
  + <details>
      <summary>Abstract</summary>
      Automatic evaluation of grammatical error correction (GEC) is essential in developing useful GEC systems. Existing methods for automatic evaluation require multiple reference sentences or manual scores. However, such resources are expensive, thereby hindering automatic evaluation for various domains and correction styles. This paper proposes an Impact-based Metric for GEC using PARAllel data, IMPARA, which utilizes correction impacts computed by parallel data comprising pairs of grammatical/ungrammatical sentences. As parallel data is cheaper than manually assessing evaluation scores, IMPARA can reduce the cost of data creation for automatic evaluation. Correlations between IMPARA and human scores indicate that IMPARA is comparable or better than existing evaluation methods. Furthermore, we find that IMPARA can perform evaluations that fit different domains and correction styles trained on various parallel data.
    </details>
+ **Multi-Perspective Document Revision**
  + Authors: Mana Ihori, Hiroshi Sato, Tomohiro Tanaka, Ryo Masumura
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.535/
  + <details>
      <summary>Abstract</summary>
      This paper presents a novel multi-perspective document revision task. In conventional studies on document revision, tasks such as grammatical error correction, sentence reordering, and discourse relation classification have been performed individually; however, these tasks simultaneously should be revised to improve the readability and clarity of a whole document. Thus, our study defines multi-perspective document revision as a task that simultaneously revises multiple perspectives. To model the task, we design a novel Japanese multi-perspective document revision dataset that simultaneously handles seven perspectives to improve the readability and clarity of a document. Although a large amount of data that simultaneously handles multiple perspectives is needed to model multi-perspective document revision elaborately, it is difficult to prepare such a large amount of this data. Therefore, our study offers a multi-perspective document revision modeling method that can use a limited amount of matched data (i.e., data for the multi-perspective document revision task) and external partially-matched data (e.g., data for the grammatical error correction task). Experiments using our created dataset demonstrate the effectiveness of using multiple partially-matched datasets to model the multi-perspective document revision task.
    </details>
+ **Position Offset Label Prediction for Grammatical Error Correction**
  + Authors: Xiuyu Wu, Jingsong Yu, Xu Sun, Yunfang Wu
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.480/
  + <details>
      <summary>Abstract</summary>
      We introduce a novel position offset label prediction subtask to the encoder-decoder architecture for grammatical error correction (GEC) task. To keep the meaning of the input sentence unchanged, only a few words should be inserted or deleted during correction, and most of tokens in the erroneous sentence appear in the paired correct sentence with limited position movement. Inspired by this observation, we design an auxiliary task to predict position offset label (POL) of tokens, which is naturally capable of integrating different correction editing operations into a unified framework. Based on the predicted POL, we further propose a new copy mechanism (P-copy) to replace the vanilla copy module. Experimental results on Chinese, English and Japanese datasets demonstrate that our proposed POL-Pc framework obviously improves the performance of baseline models. Moreover, our model yields consistent performance gain over various data augmentation methods. Especially, after incorporating synthetic data, our model achieves a 38.95 F-0.5 score on Chinese GEC dataset, which outperforms the previous state-of-the-art by a wide margin of 1.98 points.
    </details>
+ **Sequence-to-Action: Grammatical Error Correction with Action Guided Sequence Generation**
  + Authors: Jiquan Li, Junliang Guo, Yongxin Zhu, Xin Sheng, Deqiang Jiang, Bo Ren, Linli Xu
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/21345

+ **Czech Grammar Error Correction with a Large and Diverse Corpus**
  + Authors: Jakub Náplava, Milan Straka, Jana Straková, Alexandr Rosen
  + Conference: TACL
  + Link: https://aclanthology.org/2022.tacl-1.26/
  + <details>
      <summary>Abstract</summary>
      We introduce a large and diverse Czech corpus annotated for grammatical error correction (GEC) with the aim to contribute to the still scarce data resources in this domain for languages other than English. The Grammar Error Correction Corpus for Czech (GECCC) offers a variety of four domains, covering error distributions ranging from high error density essays written by non-native speakers, to website texts, where errors are expected to be much less common. We compare several Czech GEC systems, including several Transformer-based ones, setting a strong baseline to future research. Finally, we meta-evaluate common GEC metrics against human judgments on our data. We make the new Czech GEC corpus publicly available under the CC BY-SA 4.0 license at http://hdl.handle.net/11234/1-4639.
    </details>
+ **Automatic Classification of Russian Learner Errors**
  + Authors: Alla Rozovskaya
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.605/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction systems are typically evaluated overall, without taking into consideration performance on individual error types because system output is not annotated with respect to error type. We introduce a tool that automatically classifies errors in Russian learner texts. The tool takes an edit pair consisting of the original token(s) and the corresponding replacement and provides a grammatical error category. Manual evaluation of the output reveals that in more than 93% of cases the error categories are judged as correct or acceptable. We apply the tool to carry out a fine-grained evaluation on the performance of two error correction systems for Russian.
    </details>
+ **Construction of a Quality Estimation Dataset for Automatic Evaluation of Japanese Grammatical Error Correction**
  + Authors: Daisuke Suzuki, Yujin Takahashi, Ikumi Yamashita, Taichi Aida, Tosho Hirasawa, Michitaka Nakatsuji, Masato Mita, Mamoru Komachi
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.596/
  + <details>
      <summary>Abstract</summary>
      In grammatical error correction (GEC), automatic evaluation is considered as an important factor for research and development of GEC systems. Previous studies on automatic evaluation have shown that quality estimation models built from datasets with manual evaluation can achieve high performance in automatic evaluation of English GEC. However, quality estimation models have not yet been studied in Japanese, because there are no datasets for constructing quality estimation models. In this study, therefore, we created a quality estimation dataset with manual evaluation to build an automatic evaluation model for Japanese GEC. By building a quality estimation model using this dataset and conducting a meta-evaluation, we verified the usefulness of the quality estimation model for Japanese GEC.
    </details>
+ **Developing a Spell and Grammar Checker for Icelandic using an Error Corpus**
  + Authors: Hulda Óladóttir, Þórunn Arnardóttir, Anton Ingason, Vilhjálmur Þorsteinsson
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.496/
  + <details>
      <summary>Abstract</summary>
      A lack of datasets for spelling and grammatical error correction in Icelandic, along with language-specific issues, has caused a dearth of spell and grammar checking systems for the language. We present the first open-source spell and grammar checking tool for Icelandic, using an error corpus at all stages. This error corpus was in part created to aid in the development of the tool. The system is built with a rule-based tool stack comprising a tokenizer, a morphological tagger, and a parser. For token-level error annotation, tokenization rules, word lists, and a trigram model are used in error detection and correction. For sentence-level error annotation, we use specific error grammar rules in the parser as well as regex-like patterns to search syntax trees. The error corpus gives valuable insight into the errors typically made when Icelandic text is written, and guided each development phase in a test-driven manner. We assess the system’s performance with both automatic and human evaluation, using the test set in the error corpus as a reference in the automatic evaluation. The data in the error corpus development set proved useful in various ways for error detection and correction.
    </details>
+ **Dynamic Negative Example Construction for Grammatical Error Correction using Contrastive Learning**
  + Authors: He Junyi, Zhuang Junbin, Li Xia
  + Conference: CCL
  + Link: https://aclanthology.org/2022.ccl-1.83/
  + <details>
      <summary>Abstract</summary>
      “Grammatical error correction (GEC) aims at correcting texts with different types of grammatical errors into natural and correct forms. Due to the difference of error type distribution and error density, current grammatical error correction systems may over-correct writings and produce a low precision. To address this issue, in this paper, we propose a dynamic negative example construction method for grammatical error correction using contrastive learning. The proposed method can construct sufficient negative examples with diverse grammatical errors, and can be dynamically used during model training. The constructed negative examples are beneficial for the GEC model to correct sentences precisely and suppress the model from over-correction. Experimental results show that our proposed method enhances model precision, proving the effectiveness of our method.”
    </details>
+ **Enriching Grammatical Error Correction Resources for Modern Greek**
  + Authors: Katerina Korre, John Pavlopoulos
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.532/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC), a task of Natural Language Processing (NLP), is challenging for underepresented languages. This issue is most prominent in languages other than English. This paper addresses the issue of data and system sparsity for GEC purposes in the modern Greek Language. Following the most popular current approaches in GEC, we develop and test an MT5 multilingual text-to-text transformer for Greek. To our knowledge this the first attempt to create a fully-fledged GEC model for Greek. Our evaluation shows that our system reaches up to 52.63% F0.5 score on part of the Greek Native Corpus (GNC), which is 16% below the winning system of the BEA-19 shared task on English GEC. In addition, we provide an extended version of the Greek Learner Corpus (GLC), on which our model reaches up to 22.76% F0.5. Previous versions did not include corrections with the annotations which hindered the potential development of efficient GEC systems. For that reason we provide a new set of corrections. This new dataset facilitates an exploration of the generalisation abilities and robustness of our system, given that the assessment is conducted on learner data while the training on native data.
    </details>
+ **FreeTalky: Don’t Be Afraid! Conversations Made Easier by a Humanoid Robot using Persona-based Dialogue**
  + Authors: Chanjun Park, Yoonna Jang, Seolhwa Lee, Sungjin Park, Heuiseok Lim
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.132/
  + <details>
      <summary>Abstract</summary>
      We propose a deep learning-based foreign language learning platform, named FreeTalky, for people who experience anxiety dealing with foreign languages, by employing a humanoid robot NAO and various deep learning models. A persona-based dialogue system that is embedded in NAO provides an interesting and consistent multi-turn dialogue for users. Also, an grammar error correction system promotes improvement in grammar skills of the users. Thus, our system enables personalized learning based on persona dialogue and facilitates grammar learning of a user using grammar error feedback. Furthermore, we verified whether FreeTalky provides practical help in alleviating xenoglossophobia by replacing the real human in the conversation with a NAO robot, through human evaluation.
    </details>
+ **Grammatical Error Correction Systems for Automated Assessment: Are They Susceptible to Universal Adversarial Attacks?**
  + Authors: Vyas Raina, Yiting Lu, Mark Gales
  + Conference: AACL
  + Link: https://aclanthology.org/2022.aacl-main.13/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) systems are a useful tool for assessing a learner’s writing ability. These systems allow the grammatical proficiency of a candidate’s text to be assessed without requiring an examiner or teacher to read the text. A simple summary of a candidate’s ability can be measured by the total number of edits between the input text and the GEC system output: the fewer the edits the better the candidate. With advances in deep learning, GEC systems have become increasingly powerful and accurate. However, deep learning systems are susceptible to adversarial attacks, in which a small change at the input can cause large, undesired changes at the output. In the context of GEC for automated assessment, the aim of an attack can be to deceive the system into not correcting (concealing) grammatical errors to create the perception of higher language ability. An interesting aspect of adversarial attacks in this scenario is that the attack needs to be simple as it must be applied by, for example, a learner of English. The form of realistic attack examined in this work is appending the same phrase to each input sentence: a concatenative universal attack. The candidate only needs to learn a single attack phrase. State-of-the-art GEC systems are found to be susceptible to this form of simple attack, which transfers to different test sets as well as system architectures,
    </details>
+ **Improving Grammatical Error Correction for Multiword Expressions**
  + Authors: Shiva Taslimipoor, Christopher Bryant, Zheng Yuan
  + Conference: LREC
  + Link: https://aclanthology.org/2022.mwe-1.4/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is the task of automatically correcting errors in text. It has mainly been developed to assist language learning, but can also be applied to native text. This paper reports on preliminary work in improving GEC for multiword expression (MWE) error correction. We propose two systems which incorporate MWE information in two different ways: one is a multi-encoder decoder system which encodes MWE tags in a second encoder, and the other is a BART pre-trained transformer-based system that encodes MWE representations using special tokens. We show improvements in correcting specific types of verbal MWEs based on a modified version of a standard GEC evaluation approach.
    </details>
+ **MTee: Open Machine Translation Platform for Estonian Government**
  + Authors: Toms Bergmanis, Marcis Pinnis, Roberts Rozis, Jānis Šlapiņš, Valters Šics, Berta Bernāne, Guntars Pužulis, Endijs Titomers, Andre Tättar, Taido Purason, Hele-Andra Kuulmets, Agnes Luhtaru, Liisa Rätsep, Maali Tars, Annika Laumets-Tättar, Mark Fishel
  + Conference: EAMT
  + Link: https://aclanthology.org/2022.eamt-1.44/
  + <details>
      <summary>Abstract</summary>
      We present the MTee project - a research initiative funded via an Estonian public procurement to develop machine translation technology that is open-source and free of charge. The MTee project delivered an open-source platform serving state-of-the-art machine translation systems supporting four domains for six language pairs translating from Estonian into English, German, and Russian and vice-versa. The platform also features grammatical error correction and speech translation for Estonian and allows for formatted document translation and automatic domain detection. The software, data and training workflows for machine translation engines are all made publicly available for further use and research.
    </details>
+ **ProQE: Proficiency-wise Quality Estimation dataset for Grammatical Error Correction**
  + Authors: Yujin Takahashi, Masahiro Kaneko, Masato Mita, Mamoru Komachi
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.644/
  + <details>
      <summary>Abstract</summary>
      This study investigates how supervised quality estimation (QE) models of grammatical error correction (GEC) are affected by the learners’ proficiency with the data. QE models for GEC evaluations in prior work have obtained a high correlation with manual evaluations. However, when functioning in a real-world context, the data used for the reported results have limitations because prior works were biased toward data by learners with relatively high proficiency levels. To address this issue, we created a QE dataset that includes multiple proficiency levels and explored the necessity of performing proficiency-wise evaluation for QE of GEC. Our experiments demonstrated that differences in evaluation dataset proficiency affect the performance of QE models, and proficiency-wise evaluation helps create more robust models.
    </details>
+ **Semi-automatically Annotated Learner Corpus for Russian**
  + Authors: Anisia Katinskaia, Maria Lebedeva, Jue Hou, Roman Yangarber
  + Conference: LREC
  + Link: https://aclanthology.org/2022.lrec-1.88/
  + <details>
      <summary>Abstract</summary>
      We present ReLCo— the Revita Learner Corpus—a new semi-automatically annotated learner corpus for Russian. The corpus was collected while several thousand L2 learners were performing exercises using the Revita language-learning system. All errors were detected automatically by the system and annotated by type. Part of the corpus was annotated manually—this part was created for further experiments on automatic assessment of grammatical correctness. The Learner Corpus provides valuable data for studying patterns of grammatical errors, experimenting with grammatical error detection and grammatical error correction, and developing new exercises for language learners. Automating the collection and annotation makes the process of building the learner corpus much cheaper and faster, in contrast to the traditional approach of building learner corpora. We make the data publicly available.
    </details>
## 2021
+ **A Simple Recipe for Multilingual Grammatical Error Correction**
  + Authors: Sascha Rothe, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn
  + Conference: ACL
  + Link: https://aclanthology.org/2021.acl-short.89/
  + <details>
      <summary>Abstract</summary>
      This paper presents a simple recipe to trainstate-of-the-art multilingual Grammatical Error Correction (GEC) models. We achieve this by first proposing a language-agnostic method to generate a large number of synthetic examples. The second ingredient is to use large-scale multilingual language models (up to 11B parameters). Once fine-tuned on language-specific supervised sets we surpass the previous state-of-the-art results on GEC benchmarks in four languages: English, Czech, German and Russian. Having established a new set of baselines for GEC, we make our results easily reproducible and accessible by releasing a CLANG-8 dataset. It is produced by using our best model, which we call gT5, to clean the targets of a widely used yet noisy Lang-8 dataset. cLang-8 greatly simplifies typical GEC training pipelines composed of multiple fine-tuning stages – we demonstrate that performing a single fine-tuning stepon cLang-8 with the off-the-shelf language models yields further accuracy improvements over an already top-performing gT5 model for English.
    </details>
+ **A Study of Morphological Robustness of Neural Machine Translation**
  + Authors: Sai Muralidhar Jayanthi, Adithya Pratapa
  + Conference: ACL
  + Link: https://aclanthology.org/2021.sigmorphon-1.6/
  + <details>
      <summary>Abstract</summary>
      In this work, we analyze the robustness of neural machine translation systems towards grammatical perturbations in the source. In particular, we focus on morphological inflection related perturbations. While this has been recently studied for English→French (MORPHEUS) (Tan et al., 2020), it is unclear how this extends to Any→English translation systems. We propose MORPHEUS-MULTILINGUAL that utilizes UniMorph dictionaries to identify morphological perturbations to source that adversely affect the translation models. Along with an analysis of state-of-the-art pretrained MT systems, we train and analyze systems for 11 language pairs using the multilingual TED corpus (Qi et al., 2018). We also compare this to actual errors of non-native speakers using Grammatical Error Correction datasets. Finally, we present a qualitative and quantitative analysis of the robustness of Any→English translation systems.
    </details>
+ **Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**
  + Authors: Xin Sun, Tao Ge, Furu Wei, Houfeng Wang
  + Conference: ACL
  + Link: https://aclanthology.org/2021.acl-long.462/
  + <details>
      <summary>Abstract</summary>
      In this paper, we propose Shallow Aggressive Decoding (SAD) to improve the online inference efficiency of the Transformer for instantaneous Grammatical Error Correction (GEC). SAD optimizes the online inference efficiency for GEC by two innovations: 1) it aggressively decodes as many tokens as possible in parallel instead of always decoding only one token in each step to improve computational parallelism; 2) it uses a shallow decoder instead of the conventional Transformer architecture with balanced encoder-decoder depth to reduce the computational cost during inference. Experiments in both English and Chinese GEC benchmarks show that aggressive decoding could yield identical predictions to greedy decoding but with significant speedup for online inference. Its combination with the shallow decoder could offer an even higher online inference speedup over the powerful Transformer baseline without quality loss. Not only does our approach allow a single model to achieve the state-of-the-art results in English GEC benchmarks: 66.4 F0.5 in the CoNLL-14 and 72.9 F0.5 in the BEA-19 test set with an almost 10x online inference speedup over the Transformer-big model, but also it is easily adapted to other languages. Our code is available at https://github.com/AutoTemp/Shallow-Aggressive-Decoding.
    </details>
+ **Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction**
  + Authors: Piji Li, Shuming Shi
  + Conference: ACL
  + Link: https://aclanthology.org/2021.acl-long.385/
  + <details>
      <summary>Abstract</summary>
      We investigate the problem of Chinese Grammatical Error Correction (CGEC) and present a new framework named Tail-to-Tail (TtT) non-autoregressive sequence prediction to address the deep issues hidden in CGEC. Considering that most tokens are correct and can be conveyed directly from source to target, and the error positions can be estimated and corrected based on the bidirectional context information, thus we employ a BERT-initialized Transformer Encoder as the backbone model to conduct information modeling and conveying. Considering that only relying on the same position substitution cannot handle the variable-length correction cases, various operations such substitution, deletion, insertion, and local paraphrasing are required jointly. Therefore, a Conditional Random Fields (CRF) layer is stacked on the up tail to conduct non-autoregressive sequence prediction by modeling the token dependencies. Since most tokens are correct and easily to be predicted/conveyed to the target, then the models may suffer from a severe class imbalance issue. To alleviate this problem, focal loss penalty strategies are integrated into the loss functions. Moreover, besides the typical fix-length error correction datasets, we also construct a variable-length corpus to conduct experiments. Experimental results on standard datasets, especially on the variable-length datasets, demonstrate the effectiveness of TtT in terms of sentence-level Accuracy, Precision, Recall, and F1-Measure on tasks of error Detection and Correction.
    </details>
+ **Do Grammatical Error Correction Models Realize Grammatical Generalization?**
  + Authors: Masato Mita, Hitomi Yanaka
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2021.findings-acl.399/

+ **Grammatical Error Correction as GAN-like Sequence Labeling**
  + Authors: Kevin Parnow, Zuchao Li, Hai Zhao
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2021.findings-acl.290/

+ **New Dataset and Strong Baselines for the Grammatical Error Correction of Russian**
  + Authors: Viet Anh Trinh, Alla Rozovskaya
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2021.findings-acl.359/

+ **Improving Sequence-to-Sequence Pre-training via Sequence Span Rewriting**
  + Authors: Wangchunshu Zhou, Tao Ge, Canwen Xu, Ke Xu, Furu Wei
  + Conference: EMNLP
  + Link: https://aclanthology.org/2021.emnlp-main.45/
  + <details>
      <summary>Abstract</summary>
      In this paper, we propose Sequence Span Rewriting (SSR), a self-supervised task for sequence-to-sequence (Seq2Seq) pre-training. SSR learns to refine the machine-generated imperfect text spans into ground truth text. SSR provides more fine-grained and informative supervision in addition to the original text-infilling objective. Compared to the prevalent text infilling objectives for Seq2Seq pre-training, SSR is naturally more consistent with many downstream generation tasks that require sentence rewriting (e.g., text summarization, question generation, grammatical error correction, and paraphrase generation). We conduct extensive experiments by using SSR to improve the typical Seq2Seq pre-trained model T5 in a continual pre-training setting and show substantial improvements over T5 on various natural language generation tasks.
    </details>
+ **Is this the end of the gold standard? A straightforward reference-less grammatical error correction metric**
  + Authors: Md Asadul Islam, Enrico Magnani
  + Conference: EMNLP
  + Link: https://aclanthology.org/2021.emnlp-main.239/
  + <details>
      <summary>Abstract</summary>
      It is difficult to rank and evaluate the performance of grammatical error correction (GEC) systems, as a sentence can be rewritten in numerous correct ways. A number of GEC metrics have been used to evaluate proposed GEC systems; however, each system relies on either a comparison with one or more reference texts—in what is known as the gold standard for reference-based metrics—or a separate annotated dataset to fine-tune the reference-less metric. Reference-based systems have a low correlation with human judgement, cannot capture all the ways in which a sentence can be corrected, and require substantial work to develop a test dataset. We propose a reference-less GEC evaluation system that is strongly correlated with human judgement, solves the issues related to the use of a reference, and does not need another annotated dataset for fine-tuning. The proposed system relies solely on commonly available tools. Additionally, currently available reference-less metrics do not work properly when part of a sentence is repeated as opposed to reference-based metrics. In our proposed system, we look to address issues inherent in reference-less metrics and reference-based metrics.
    </details>
+ **LM-Critic: Language Models for Unsupervised Grammatical Error Correction**
  + Authors: Michihiro Yasunaga, Jure Leskovec, Percy Liang
  + Conference: EMNLP
  + Link: https://aclanthology.org/2021.emnlp-main.611/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) requires a set of labeled ungrammatical / grammatical sentence pairs for training, but obtaining such annotation can be prohibitively expensive. Recently, the Break-It-Fix-It (BIFI) framework has demonstrated strong results on learning to repair a broken program without any labeled examples, but this relies on a perfect critic (e.g., a compiler) that returns whether an example is valid or not, which does not exist for the GEC task. In this work, we show how to leverage a pretrained language model (LM) in defining an LM-Critic, which judges a sentence to be grammatical if the LM assigns it a higher probability than its local perturbations. We apply this LM-Critic and BIFI along with a large set of unlabeled sentences to bootstrap realistic ungrammatical / grammatical pairs for training a corrector. We evaluate our approach on GEC datasets on multiple domains (CoNLL-2014, BEA-2019, GMEG-wiki and GMEG-yahoo) and show that it outperforms existing methods in both the unsupervised setting (+7.7 F0.5) and the supervised setting (+0.5 F0.5).
    </details>
+ **MiSS: An Assistant for Multi-Style Simultaneous Translation**
  + Authors: Zuchao Li, Kevin Parnow, Masao Utiyama, Eiichiro Sumita, Hai Zhao
  + Conference: EMNLP
  + Link: https://aclanthology.org/2021.emnlp-demo.1/
  + <details>
      <summary>Abstract</summary>
      In this paper, we present MiSS, an assistant for multi-style simultaneous translation. Our proposed translation system has five key features: highly accurate translation, simultaneous translation, translation for multiple text styles, back-translation for translation quality evaluation, and grammatical error correction. With this system, we aim to provide a complete translation experience for machine translation users. Our design goals are high translation accuracy, real-time translation, flexibility, and measurable translation quality. Compared with the free commercial translation systems commonly used, our translation assistance system regards the machine translation application as a more complete and fully-featured tool for users. By incorporating additional features and giving the user better control over their experience, we improve translation efficiency and performance. Additionally, our assistant system combines machine translation, grammatical error correction, and interactive edits, and uses a crowdsourcing mode to collect more data for further training to improve both the machine translation and grammatical error correction models. A short video demonstrating our system is available at https://www.youtube.com/watch?v=ZGCo7KtRKd8.
    </details>
+ **Multi-Class Grammatical Error Detection for Correction: A Tale of Two Systems**
  + Authors: Zheng Yuan, Shiva Taslimipoor, Christopher Davis, Christopher Bryant
  + Conference: EMNLP
  + Link: https://aclanthology.org/2021.emnlp-main.687/
  + <details>
      <summary>Abstract</summary>
      In this paper, we show how a multi-class grammatical error detection (GED) system can be used to improve grammatical error correction (GEC) for English. Specifically, we first develop a new state-of-the-art binary detection system based on pre-trained ELECTRA, and then extend it to multi-class detection using different error type tagsets derived from the ERRANT framework. Output from this detection system is used as auxiliary input to fine-tune a novel encoder-decoder GEC model, and we subsequently re-rank the N-best GEC output to find the hypothesis that most agrees with the GED output. Results show that fine-tuning the GEC system using 4-class GED produces the best model, but re-ranking using 55-class GED leads to the best performance overall. This suggests that different multi-class GED systems benefit GEC in different ways. Ultimately, our system outperforms all other previous work that combines GED and GEC, and achieves a new single-model NMT-based state of the art on the BEA-test benchmark.
    </details>
+ **An Alignment-Agnostic Model for Chinese Text Error Correction**
  + Authors: Liying Zheng, Yue Deng, Weishun Song, Liang Xu, Jing Xiao
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2021.findings-emnlp.30/
  + <details>
      <summary>Abstract</summary>
      This paper investigates how to correct Chinese text errors with types of mistaken, missing and redundant characters, which are common for Chinese native speakers. Most existing models based on detect-correct framework can correct mistaken characters, but cannot handle missing or redundant characters due to inconsistency between model inputs and outputs. Although Seq2Seq-based or sequence tagging methods provide solutions to the three error types and achieved relatively good results in English context, they do not perform well in Chinese context according to our experiments. In our work, we propose a novel alignment-agnostic detect-correct framework that can handle both text aligned and non-aligned situations and can serve as a cold start model when no annotation data are provided. Experimental results on three datasets demonstrate that our method is effective and achieves a better performance than most recent published models.
    </details>
+ **Beyond Grammatical Error Correction: Improving L1-influenced research writing in English using pre-trained encoder-decoder models**
  + Authors: Gustavo Zomer, Ana Frankenberg-Garcia
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2021.findings-emnlp.216/
  + <details>
      <summary>Abstract</summary>
      In this paper, we present a new method for training a writing improvement model adapted to the writer’s first language (L1) that goes beyond grammatical error correction (GEC). Without using annotated training data, we rely solely on pre-trained language models fine-tuned with parallel corpora of reference translation aligned with machine translation. We evaluate our model with corpora of academic papers written in English by L1 Portuguese and L1 Spanish scholars and a reference corpus of expert academic English. We show that our model is able to address specific L1-influenced writing and more complex linguistic phenomena than existing methods, outperforming what a state-of-the-art GEC system can achieve in this regard. Our code and data are open to other researchers.
    </details>
+ **Grammatical Error Correction with Contrastive Learning in Low Error Density Domains**
  + Authors: Hannan Cao, Wenmian Yang, Hwee Tou Ng
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2021.findings-emnlp.419/
  + <details>
      <summary>Abstract</summary>
      Although grammatical error correction (GEC) has achieved good performance on texts written by learners of English as a second language, performance on low error density domains where texts are written by English speakers of varying levels of proficiency can still be improved. In this paper, we propose a contrastive learning approach to encourage the GEC model to assign a higher probability to a correct sentence while reducing the probability of incorrect sentences that the model tends to generate, so as to improve the accuracy of the model. Experimental results show that our approach significantly improves the performance of GEC models in low error density domains, when evaluated on the benchmark CWEB dataset.
    </details>
+ **Comparison of Grammatical Error Correction Using Back-Translation Models**
  + Authors: Aomi Koyama, Kengo Hotate, Masahiro Kaneko, Mamoru Komachi
  + Conference: NAACL
  + Link: https://aclanthology.org/2021.naacl-srw.16/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) suffers from a lack of sufficient parallel data. Studies on GEC have proposed several methods to generate pseudo data, which comprise pairs of grammatical and artificially produced ungrammatical sentences. Currently, a mainstream approach to generate pseudo data is back-translation (BT). Most previous studies using BT have employed the same architecture for both the GEC and BT models. However, GEC models have different correction tendencies depending on the architecture of their models. Thus, in this study, we compare the correction tendencies of GEC models trained on pseudo data generated by three BT models with different architectures, namely, Transformer, CNN, and LSTM. The results confirm that the correction tendencies for each error type are different for every BT model. In addition, we investigate the correction tendencies when using a combination of pseudo data generated by different BT models. As a result, we find that the combination of different BT models improves or interpolates the performance of each error type compared with using a single BT model with different seeds.
    </details>
+ **Negative language transfer in learner English: A new dataset**
  + Authors: Leticia Farias Wanderley, Nicole Zhao, Carrie Demmans Epp
  + Conference: NAACL
  + Link: https://aclanthology.org/2021.naacl-main.251/
  + <details>
      <summary>Abstract</summary>
      Automatic personalized corrective feedback can help language learners from different backgrounds better acquire a new language. This paper introduces a learner English dataset in which learner errors are accompanied by information about possible error sources. This dataset contains manually annotated error causes for learner writing errors. These causes tie learner mistakes to structures from their first languages, when the rules in English and in the first language diverge. This new dataset will enable second language acquisition researchers to computationally analyze a large quantity of learner errors that are related to language transfer from the learners’ first language. The dataset can also be applied in personalizing grammatical error correction systems according to the learners’ first language and in providing feedback that is informed by the cause of an error.
    </details>
+ **Neural Quality Estimation with Multiple Hypotheses for Grammatical Error Correction**
  + Authors: Zhenghao Liu, Xiaoyuan Yi, Maosong Sun, Liner Yang, Tat-Seng Chua
  + Conference: NAACL
  + Link: https://aclanthology.org/2021.naacl-main.429/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) aims to correct writing errors and help language learners improve their writing skills. However, existing GEC models tend to produce spurious corrections or fail to detect lots of errors. The quality estimation model is necessary to ensure learners get accurate GEC results and avoid misleading from poorly corrected sentences. Well-trained GEC models can generate several high-quality hypotheses through decoding, such as beam search, which provide valuable GEC evidence and can be used to evaluate GEC quality. However, existing models neglect the possible GEC evidence from different hypotheses. This paper presents the Neural Verification Network (VERNet) for GEC quality estimation with multiple hypotheses. VERNet establishes interactions among hypotheses with a reasoning graph and conducts two kinds of attention mechanisms to propagate GEC evidence to verify the quality of generated hypotheses. Our experiments on four GEC datasets show that VERNet achieves state-of-the-art grammatical error detection performance, achieves the best quality estimation results, and significantly improves GEC performance by reranking hypotheses. All data and source codes are available at https://github.com/thunlp/VERNet.
    </details>
+ **Data Strategies for Low-Resource Grammatical Error Correction**
  + Authors: Simon Flachs, Felix Stahlberg, Shankar Kumar
  + Conference: EACL
  + Link: https://aclanthology.org/2021.bea-1.12/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is a task that has been extensively investigated for the English language. However, for low-resource languages the best practices for training GEC systems have not yet been systematically determined. We investigate how best to take advantage of existing data sources for improving GEC systems for languages with limited quantities of high quality training data. We show that methods for generating artificial training data for GEC can benefit from including morphological errors. We also demonstrate that noisy error correction data gathered from Wikipedia revision histories and the language learning website Lang8, are valuable data sources. Finally, we show that GEC systems pre-trained on noisy data sources can be fine-tuned effectively using small amounts of high quality, human-annotated data.
    </details>
+ **Document-level grammatical error correction**
  + Authors: Zheng Yuan, Christopher Bryant
  + Conference: EACL
  + Link: https://aclanthology.org/2021.bea-1.8/
  + <details>
      <summary>Abstract</summary>
      Document-level context can provide valuable information in grammatical error correction (GEC), which is crucial for correcting certain errors and resolving inconsistencies. In this paper, we investigate context-aware approaches and propose document-level GEC systems. Additionally, we employ a three-step training strategy to benefit from both sentence-level and document-level data. Our system outperforms previous document-level and all other NMT-based single-model systems, achieving state of the art on a common test set.
    </details>
+ **How Good (really) are Grammatical Error Correction Systems?**
  + Authors: Alla Rozovskaya, Dan Roth
  + Conference: EACL
  + Link: https://aclanthology.org/2021.eacl-main.231/
  + <details>
      <summary>Abstract</summary>
      Standard evaluations of Grammatical Error Correction (GEC) systems make use of a fixed reference text generated relative to the original text; they show, even when using multiple references, that we have a long way to go. This analysis paper studies the performance of GEC systems relative to closest-gold – a gold reference text created relative to the output of a system. Surprisingly, we show that the real performance is 20-40 points better than standard evaluations show. Moreover, the performance remains high even when considering any of the top-10 hypotheses produced by a system. Importantly, the type of mistakes corrected by lower-ranked hypotheses differs in interesting ways from the top one, providing an opportunity to focus on a range of errors – local spelling and grammar edits vs. more complex lexical improvements. Our study shows these results in English and Russian, and thus provides a preliminary proposal for a more realistic evaluation of GEC systems.
    </details>
+ **Synthetic Data Generation for Grammatical Error Correction with Tagged Corruption Models**
  + Authors: Felix Stahlberg, Shankar Kumar
  + Conference: EACL
  + Link: https://aclanthology.org/2021.bea-1.4/
  + <details>
      <summary>Abstract</summary>
      Synthetic data generation is widely known to boost the accuracy of neural grammatical error correction (GEC) systems, but existing methods often lack diversity or are too simplistic to generate the broad range of grammatical errors made by human writers. In this work, we use error type tags from automatic annotation tools such as ERRANT to guide synthetic data generation. We compare several models that can produce an ungrammatical sentence given a clean sentence and an error type tag. We use these models to build a new, large synthetic pre-training data set with error tag frequency distributions matching a given development set. Our synthetic data set yields large and consistent gains, improving the state-of-the-art on the BEA-19 and CoNLL-14 test sets. We also show that our approach is particularly effective in adapting a GEC system, trained on mixed native and non-native English, to a native English test set, even surpassing real training data consisting of high-quality sentence pairs.
    </details>
+ **Automatic Error Type Annotation for Arabic**
  + Authors: Riadh Belkebir, Nizar Habash
  + Conference: CoNLL
  + Link: https://aclanthology.org/2021.conll-1.47/
  + <details>
      <summary>Abstract</summary>
      We present ARETA, an automatic error type annotation system for Modern Standard Arabic. We design ARETA to address Arabic’s morphological richness and orthographic ambiguity. We base our error taxonomy on the Arabic Learner Corpus (ALC) Error Tagset with some modifications. ARETA achieves a performance of 85.8% (micro average F1 score) on a manually annotated blind test portion of ALC. We also demonstrate ARETA’s usability by applying it to a number of submissions from the QALB 2014 shared task for Arabic grammatical error correction. The resulting analyses give helpful insights on the strengths and weaknesses of different submissions, which is more useful than the opaque M2 scoring metrics used in the shared task. ARETA employs a large Arabic morphological analyzer, but is completely unsupervised otherwise. We make ARETA publicly available.
    </details>
+ **Data Augmentation of Incorporating Real Error Patterns and Linguistic Knowledge for Grammatical Error Correction**
  + Authors: Xia Li, Junyi He
  + Conference: CoNLL
  + Link: https://aclanthology.org/2021.conll-1.17/
  + <details>
      <summary>Abstract</summary>
      Data augmentation aims at expanding training data with clean text using noising schemes to improve the performance of grammatical error correction (GEC). In practice, there are a great number of real error patterns in the manually annotated training data. We argue that these real error patterns can be introduced into clean text to effectively generate more real and high quality synthetic data, which is not fully explored by previous studies. Moreover, we also find that linguistic knowledge can be incorporated into data augmentation for generating more representative and more diverse synthetic data. In this paper, we propose a novel data augmentation method that fully considers the real error patterns and the linguistic knowledge for the GEC task. We conduct extensive experiments on public data sets and the experimental results show that our method outperforms several strong baselines with far less external unlabeled clean text data, highlighting its extraordinary effectiveness in the GEC task that lacks large-scale labeled training data.
    </details>
+ **Automatic Extraction of English Grammar Pattern Correction Rules**
  + Authors: Kuan-Yu Shen, Yi-Chien Lin, Jason S. Chang
  + Conference: ROCLING
  + Link: https://aclanthology.org/2021.rocling-1.32/
  + <details>
      <summary>Abstract</summary>
      We introduce a method for generating error-correction rules for grammar pattern errors in a given annotated learner corpus. In our approach, annotated edits in the learner corpus are converted into edit rules for correcting common writing errors. The method involves automatic extraction of grammar patterns, and automatic alignment of the erroneous patterns and correct patterns. At run-time, grammar patterns are extracted from the grammatically correct sentences, and correction rules are retrieved by aligning the extracted grammar patterns with the erroneous patterns. Using the proposed method, we generate 1,499 high-quality correction rules related to 232 headwords. The method can be used to assist ESL students in avoiding grammatical errors, and aid teachers in correcting students’ essays. Additionally, the method can be used in the compilation of collocation error dictionaries and the construction of grammar error correction systems.
    </details>
+ **GECko+: a Grammatical and Discourse Error Correction Tool**
  + Authors: Eduardo Calò, Léo Jacqmin, Thibo Rosemplatt, Maxime Amblard, Miguel Couceiro, Ajinkya Kulkarni
  + Conference: JEP/TALN/RECITAL
  + Link: https://aclanthology.org/2021.jeptalnrecital-demo.3/
  + <details>
      <summary>Abstract</summary>
      GECko+ : a Grammatical and Discourse Error Correction Tool We introduce GECko+, a web-based writing assistance tool for English that corrects errors both at the sentence and at the discourse level. It is based on two state-of-the-art models for grammar error correction and sentence ordering. GECko+ is available online as a web application that implements a pipeline combining the two models.
    </details>
+ **Grammatical Error Correction via Supervised Attention in the Vicinity of Errors**
  + Authors: Hiromichi Ishii, Akihiro Tamura, Takashi Ninomiya
  + Conference: PACLIC
  + Link: https://aclanthology.org/2021.paclic-1.24/

+ **Leveraging Task Information in Grammatical Error Correction for Short Answer Assessment through Context-based Reranking**
  + Authors: Ramon Ziai, Anna Karnysheva
  + Conference: WS
  + Link: https://aclanthology.org/2021.nlp4call-1.6/

+ **System Combination for Grammatical Error Correction Based on Integer Programming**
  + Authors: Ruixi Lin, Hwee Tou Ng
  + Conference: RANLP
  + Link: https://aclanthology.org/2021.ranlp-1.94/
  + <details>
      <summary>Abstract</summary>
      In this paper, we propose a system combination method for grammatical error correction (GEC), based on nonlinear integer programming (IP). Our method optimizes a novel F score objective based on error types, and combines multiple end-to-end GEC systems. The proposed IP approach optimizes the selection of a single best system for each grammatical error type present in the data. Experiments of the IP approach on combining state-of-the-art standalone GEC systems show that the combined system outperforms all standalone systems. It improves F0.5 score by 3.61% when combining the two best participating systems in the BEA 2019 shared task, and achieves F0.5 score of 73.08%. We also perform experiments to compare our IP approach with another state-of-the-art system combination method for GEC, demonstrating IP’s competitive combination capability.
    </details>
+ **Various Errors Improve Neural Grammatical Error Correction**
  + Authors: Shota Koyama, Hiroya Takamura, Naoaki Okazaki
  + Conference: PACLIC
  + Link: https://aclanthology.org/2021.paclic-1.27/

+ **基于字词粒度噪声数据增强的中文语法纠错(Chinese Grammatical Error Correction enhanced by Data Augmentation from Word and Character Levels)**
  + Authors: Zecheng Tang (汤泽成), Yixin Ji (纪一心), Yibo Zhao (赵怡博), Junhui Li (李军辉)
  + Conference: CCL
  + Link: https://aclanthology.org/2021.ccl-1.73/
  + <details>
      <summary>Abstract</summary>
      语法纠错是自然语言处理领域的热门任务之一,其目的是将错误的句子修改为正确的句子。为了缓解中文训练语料不足的问题,本文从数据增强的角度出发,提出一种新颖的扩充和增强数据的方法。具体地,为了使模型能更好地获取不同类型和不同粒度的错误,本文首先对语法纠错中出现的错误进行了字和词粒度的分类,在此基础上提出了融合字词粒度噪声的数据增强方法,以此获得大规模且质量较高的错误数据集。基于NLPCC2018共享任务的实验结果表明,本文提出的融合字词粒度加噪方法能够显著提升模型的性能,在该数据集上达到了最优的性能。最后,本文分析了错误类型和数据规模对中文语法纠错模型性能的影响。
    </details>
## 2020
+ **A Comparative Study of Synthetic Data Generation Methods for Grammatical Error Correction**
  + Authors: Max White, Alla Rozovskaya
  + Conference: ACL
  + Link: https://aclanthology.org/2020.bea-1.21/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is concerned with correcting grammatical errors in written text. Current GEC systems, namely those leveraging statistical and neural machine translation, require large quantities of annotated training data, which can be expensive or impractical to obtain. This research compares techniques for generating synthetic data utilized by the two highest scoring submissions to the restricted and low-resource tracks in the BEA-2019 Shared Task on Grammatical Error Correction.
    </details>
+ **Encoder-Decoder Models Can Benefit from Pre-trained Masked Language Models in Grammatical Error Correction**
  + Authors: Masahiro Kaneko, Masato Mita, Shun Kiyono, Jun Suzuki, Kentaro Inui
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-main.391/
  + <details>
      <summary>Abstract</summary>
      This paper investigates how to effectively incorporate a pre-trained masked language model (MLM), such as BERT, into an encoder-decoder (EncDec) model for grammatical error correction (GEC). The answer to this question is not as straightforward as one might expect because the previous common methods for incorporating a MLM into an EncDec model have potential drawbacks when applied to GEC. For example, the distribution of the inputs to a GEC model can be considerably different (erroneous, clumsy, etc.) from that of the corpora used for pre-training MLMs; however, this issue is not addressed in the previous methods. Our experiments show that our proposed method, where we first fine-tune a MLM with a given GEC corpus and then use the output of the fine-tuned MLM as additional features in the GEC model, maximizes the benefit of the MLM. The best-performing model achieves state-of-the-art performances on the BEA-2019 and CoNLL-2014 benchmarks. Our code is publicly available at: https://github.com/kanekomasahiro/bert-gec.
    </details>
+ **GECToR – Grammatical Error Correction: Tag, Not Rewrite**
  + Authors: Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, Oleksandr Skurzhanskyi
  + Conference: ACL
  + Link: https://aclanthology.org/2020.bea-1.16/
  + <details>
      <summary>Abstract</summary>
      In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an F_0.5 of 65.3/66.5 on CONLL-2014 (test) and F_0.5 of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system.
    </details>
+ **Grammatical Error Correction Using Pseudo Learner Corpus Considering Learner’s Error Tendency**
  + Authors: Yujin Takahashi, Satoru Katsumata, Mamoru Komachi
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-srw.5/
  + <details>
      <summary>Abstract</summary>
      Recently, several studies have focused on improving the performance of grammatical error correction (GEC) tasks using pseudo data. However, a large amount of pseudo data are required to train an accurate GEC model. To address the limitations of language and computational resources, we assume that introducing pseudo errors into sentences similar to those written by the language learners is more efficient, rather than incorporating random pseudo errors into monolingual data. In this regard, we study the effect of pseudo data on GEC task performance using two approaches. First, we extract sentences that are similar to the learners’ sentences from monolingual data. Second, we generate realistic pseudo errors by considering error types that learners often make. Based on our comparative results, we observe that F0.5 scores for the Russian GEC task are significantly improved.
    </details>
+ **Semantic Parsing for English as a Second Language**
  + Authors: Yuanyuan Zhao, Weiwei Sun, Junjie Cao, Xiaojun Wan
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-main.606/
  + <details>
      <summary>Abstract</summary>
      This paper is concerned with semantic parsing for English as a second language (ESL). Motivated by the theoretical emphasis on the learning challenges that occur at the syntax-semantics interface during second language acquisition, we formulate the task based on the divergence between literal and intended meanings. We combine the complementary strengths of English Resource Grammar, a linguistically-precise hand-crafted deep grammar, and TLE, an existing manually annotated ESL UD-TreeBank with a novel reranking model. Experiments demonstrate that in comparison to human annotations, our method can obtain a very promising SemBanking quality. By means of the newly created corpus, we evaluate state-of-the-art semantic parsing as well as grammatical error correction models. The evaluation profiles the performance of neural NLP techniques for handling ESL data and suggests some research directions.
    </details>
+ **Using Context in Neural Machine Translation Training Objectives**
  + Authors: Danielle Saunders, Felix Stahlberg, Bill Byrne
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-main.693/
  + <details>
      <summary>Abstract</summary>
      We present Neural Machine Translation (NMT) training using document-level metrics with batch-level documents. Previous sequence-objective approaches to NMT training focus exclusively on sentence-level metrics like sentence BLEU which do not correspond to the desired evaluation metric, typically document BLEU. Meanwhile research into document-level NMT training focuses on data or model architecture rather than training procedure. We find that each of these lines of research has a clear space in it for the other, and propose merging them with a scheme that allows a document-level evaluation metric to be used in the NMT training objective. We first sample pseudo-documents from sentence samples. We then approximate the expected document BLEU gradient with Monte Carlo sampling for use as a cost function in Minimum Risk Training (MRT). This two-level sampling procedure gives NMT performance gains over sequence MRT and maximum-likelihood training. We demonstrate that training is more robust for document-level metrics than with sequence metrics. We further demonstrate improvements on NMT with TER and Grammatical Error Correction (GEC) using GLEU, both metrics used at the document level for evaluations.
    </details>
+ **Grammatical Error Correction in Low Error Density Domains: A New Benchmark and Analyses**
  + Authors: Simon Flachs, Ophélie Lacroix, Helen Yannakoudakis, Marek Rei, Anders Søgaard
  + Conference: EMNLP
  + Link: https://aclanthology.org/2020.emnlp-main.680/
  + <details>
      <summary>Abstract</summary>
      Evaluation of grammatical error correction (GEC) systems has primarily focused on essays written by non-native learners of English, which however is only part of the full spectrum of GEC applications. We aim to broaden the target domain of GEC and release CWEB, a new benchmark for GEC consisting of website text generated by English speakers of varying levels of proficiency. Website data is a common and important domain that contains far fewer grammatical errors than learner essays, which we show presents a challenge to state-of-the-art GEC systems. We demonstrate that a factor behind this is the inability of systems to rely on a strong internal language model in low error density domains. We hope this work shall facilitate the development of open-domain GEC models that generalize to different topics and genres.
    </details>
+ **Improving Grammatical Error Correction Models with Purpose-Built Adversarial Examples**
  + Authors: Lihao Wang, Xiaoqing Zheng
  + Conference: EMNLP
  + Link: https://aclanthology.org/2020.emnlp-main.228/
  + <details>
      <summary>Abstract</summary>
      A sequence-to-sequence (seq2seq) learning with neural networks empirically shows to be an effective framework for grammatical error correction (GEC), which takes a sentence with errors as input and outputs the corrected one. However, the performance of GEC models with the seq2seq framework heavily relies on the size and quality of the corpus on hand. We propose a method inspired by adversarial training to generate more meaningful and valuable training examples by continually identifying the weak spots of a model, and to enhance the model by gradually adding the generated adversarial examples to the training set. Extensive experimental results show that such adversarial training can improve both the generalization and robustness of GEC models.
    </details>
+ **Improving the Efficiency of Grammatical Error Correction with Erroneous Span Detection and Correction**
  + Authors: Mengyun Chen, Tao Ge, Xingxing Zhang, Furu Wei, Ming Zhou
  + Conference: EMNLP
  + Link: https://aclanthology.org/2020.emnlp-main.581/
  + <details>
      <summary>Abstract</summary>
      We propose a novel language-independent approach to improve the efficiency for Grammatical Error Correction (GEC) by dividing the task into two subtasks: Erroneous Span Detection (ESD) and Erroneous Span Correction (ESC). ESD identifies grammatically incorrect text spans with an efficient sequence tagging model. Then, ESC leverages a seq2seq model to take the sentence with annotated erroneous spans as input and only outputs the corrected text for these spans. Experiments show our approach performs comparably to conventional seq2seq approaches in both English and Chinese GEC benchmarks with less than 50% time cost for inference.
    </details>
+ **Seq2Edits: Sequence Transduction Using Span-level Edit Operations**
  + Authors: Felix Stahlberg, Shankar Kumar
  + Conference: EMNLP
  + Link: https://aclanthology.org/2020.emnlp-main.418/
  + <details>
      <summary>Abstract</summary>
      We propose Seq2Edits, an open-vocabulary approach to sequence editing for natural language processing (NLP) tasks with a high degree of overlap between input and output texts. In this approach, each sequence-to-sequence transduction is represented as a sequence of edit operations, where each operation either replaces an entire source span with target tokens or keeps it unchanged. We evaluate our method on five NLP tasks (text normalization, sentence fusion, sentence splitting & rephrasing, text simplification, and grammatical error correction) and report competitive results across the board. For grammatical error correction, our method speeds up inference by up to 5.2x compared to full sequence models because inference time depends on the number of edits rather than the number of target tokens. For text normalization, sentence fusion, and grammatical error correction, our approach improves explainability by associating each edit operation with a human-readable tag.
    </details>
+ **A Self-Refinement Strategy for Noise Reduction in Grammatical Error Correction**
  + Authors: Masato Mita, Shun Kiyono, Masahiro Kaneko, Jun Suzuki, Kentaro Inui
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2020.findings-emnlp.26/
  + <details>
      <summary>Abstract</summary>
      Existing approaches for grammatical error correction (GEC) largely rely on supervised learning with manually created GEC datasets. However, there has been little focus on verifying and ensuring the quality of the datasets, and on how lower-quality data might affect GEC performance. We indeed found that there is a non-negligible amount of “noise” where errors were inappropriately edited or left uncorrected. To address this, we designed a self-refinement method where the key idea is to denoise these datasets by leveraging the prediction consistency of existing models, and outperformed strong denoising baseline methods. We further applied task-specific techniques and achieved state-of-the-art performance on the CoNLL-2014, JFLEG, and BEA-2019 benchmarks. We then analyzed the effect of the proposed denoising method, and found that our approach leads to improved coverage of corrections and facilitated fluency edits which are reflected in higher recall and overall performance.
    </details>
+ **Adversarial Grammatical Error Correction**
  + Authors: Vipul Raheja, Dimitris Alikaniotis
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2020.findings-emnlp.275/
  + <details>
      <summary>Abstract</summary>
      Recent works in Grammatical Error Correction (GEC) have leveraged the progress in Neural Machine Translation (NMT), to learn rewrites from parallel corpora of grammatically incorrect and corrected sentences, achieving state-of-the-art results. At the same time, Generative Adversarial Networks (GANs) have been successful in generating realistic texts across many different tasks by learning to directly minimize the difference between human-generated and synthetic text. In this work, we present an adversarial learning approach to GEC, using the generator-discriminator framework. The generator is a Transformer model, trained to produce grammatically correct sentences given grammatically incorrect ones. The discriminator is a sentence-pair classification model, trained to judge a given pair of grammatically incorrect-correct sentences on the quality of grammatical correction. We pre-train both the discriminator and the generator on parallel texts and then fine-tune them further using a policy gradient method that assigns high rewards to sentences which could be true corrections of the grammatically incorrect text. Experimental results on FCE, CoNLL-14, and BEA-19 datasets show that Adversarial-GEC can achieve competitive GEC quality compared to NMT-based baselines.
    </details>
+ **Improving Grammatical Error Correction with Machine Translation Pairs**
  + Authors: Wangchunshu Zhou, Tao Ge, Chang Mu, Ke Xu, Furu Wei, Ming Zhou
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2020.findings-emnlp.30/
  + <details>
      <summary>Abstract</summary>
      We propose a novel data synthesis method to generate diverse error-corrected sentence pairs for improving grammatical error correction, which is based on a pair of machine translation models (e.g., Chinese to English) of different qualities (i.e., poor and good). The poor translation model can resemble the ESL (English as a second language) learner and tends to generate translations of low quality in terms of fluency and grammaticality, while the good translation model generally generates fluent and grammatically correct translations. With the pair of translation models, we can generate unlimited numbers of poor to good English sentence pairs from text in the source language (e.g., Chinese) of the translators. Our approach can generate various error-corrected patterns and nicely complement the other data synthesis approaches for GEC. Experimental results demonstrate the data generated by our approach can effectively help a GEC model to improve the performance and achieve the state-of-the-art single-model performance in BEA-19 and CoNLL-14 benchmark datasets.
    </details>
+ **Pseudo-Bidirectional Decoding for Local Sequence Transduction**
  + Authors: Wangchunshu Zhou, Tao Ge, Ke Xu
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2020.findings-emnlp.136/
  + <details>
      <summary>Abstract</summary>
      Local sequence transduction (LST) tasks are sequence transduction tasks where there exists massive overlapping between the source and target sequences, such as grammatical error correction and spell or OCR correction. Motivated by this characteristic of LST tasks, we propose Pseudo-Bidirectional Decoding (PBD), a simple but versatile approach for LST tasks. PBD copies the representation of source tokens to the decoder as pseudo future context that enables the decoder self-attention to attends to its bi-directional context. In addition, the bidirectional decoding scheme and the characteristic of LST tasks motivate us to share the encoder and the decoder of LST models. Our approach provides right-side context information for the decoder, reduces the number of parameters by half, and provides good regularization effects. Experimental results on several benchmark datasets show that our approach consistently improves the performance of standard seq2seq models on LST tasks.
    </details>
+ **A Crash Course in Automatic Grammatical Error Correction**
  + Authors: Roman Grundkiewicz, Christopher Bryant, Mariano Felice
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-tutorials.6/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is the task of automatically detecting and correcting all types of errors in written text. Although most research has focused on correcting errors in the context of English as a Second Language (ESL), GEC can also be applied to other languages and native text. The main application of a GEC system is thus to assist humans with their writing. Academic and commercial interest in GEC has grown significantly since the Helping Our Own (HOO) and Conference on Natural Language Learning (CoNLL) shared tasks in 2011-14, and a record-breaking 24 teams took part in the recent Building Educational Applications (BEA) shared task. Given this interest, and the recent shift towards neural approaches, we believe the time is right to offer a tutorial on GEC for researchers who may be new to the field or who are interested in the current state of the art and future challenges. With this in mind, the main goal of this tutorial is not only to bring attendees up to speed with GEC in general, but also examine the development of neural-based GEC systems.
    </details>
+ **Cross-lingual Transfer Learning for Grammatical Error Correction**
  + Authors: Ikumi Yamashita, Satoru Katsumata, Masahiro Kaneko, Aizhan Imankulova, Mamoru Komachi
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.415/
  + <details>
      <summary>Abstract</summary>
      In this study, we explore cross-lingual transfer learning in grammatical error correction (GEC) tasks. Many languages lack the resources required to train GEC models. Cross-lingual transfer learning from high-resource languages (the source models) is effective for training models of low-resource languages (the target models) for various tasks. However, in GEC tasks, the possibility of transferring grammatical knowledge (e.g., grammatical functions) across languages is not evident. Therefore, we investigate cross-lingual transfer learning methods for GEC. Our results demonstrate that transfer learning from other languages can improve the accuracy of GEC. We also demonstrate that proximity to source languages has a significant impact on the accuracy of correcting certain types of errors.
    </details>
+ **Generating Diverse Corrections with Local Beam Search for Grammatical Error Correction**
  + Authors: Kengo Hotate, Masahiro Kaneko, Mamoru Komachi
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.193/
  + <details>
      <summary>Abstract</summary>
      In this study, we propose a beam search method to obtain diverse outputs in a local sequence transduction task where most of the tokens in the source and target sentences overlap, such as in grammatical error correction (GEC). In GEC, it is advisable to rewrite only the local sequences that must be rewritten while leaving the correct sequences unchanged. However, existing methods of acquiring various outputs focus on revising all tokens of a sentence. Therefore, existing methods may either generate ungrammatical sentences because they force the entire sentence to be changed or produce non-diversified sentences by weakening the constraints to avoid generating ungrammatical sentences. Considering these issues, we propose a method that does not rewrite all the tokens in a text, but only rewrites those parts that need to be diversely corrected. Our beam search method adjusts the search token in the beam according to the probability that the prediction is copied from the source sentence. The experimental results show that our proposed method generates more diverse corrections than existing methods without losing accuracy in the GEC task.
    </details>
+ **Grammatical error detection in transcriptions of spoken English**
  + Authors: Andrew Caines, Christian Bentz, Kate Knill, Marek Rei, Paula Buttery
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.195/
  + <details>
      <summary>Abstract</summary>
      We describe the collection of transcription corrections and grammatical error annotations for the CrowdED Corpus of spoken English monologues on business topics. The corpus recordings were crowdsourced from native speakers of English and learners of English with German as their first language. The new transcriptions and annotations are obtained from different crowdworkers: we analyse the 1108 new crowdworker submissions and propose that they can be used for automatic transcription post-editing and grammatical error correction for speech. To further explore the data we train grammatical error detection models with various configurations including pre-trained and contextual word representations as input, additional features and auxiliary objectives, and extra training data from written error-annotated corpora. We find that a model concatenating pre-trained and contextual word representations as input performs best, and that additional information does not lead to further performance gains.
    </details>
+ **Heterogeneous Recycle Generation for Chinese Grammatical Error Correction**
  + Authors: Charles Hinson, Hen-Hsen Huang, Hsin-Hsi Chen
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.199/
  + <details>
      <summary>Abstract</summary>
      Most recent works in the field of grammatical error correction (GEC) rely on neural machine translation-based models. Although these models boast impressive performance, they require a massive amount of data to properly train. Furthermore, NMT-based systems treat GEC purely as a translation task and overlook the editing aspect of it. In this work we propose a heterogeneous approach to Chinese GEC, composed of a NMT-based model, a sequence editing model, and a spell checker. Our methodology not only achieves a new state-of-the-art performance for Chinese GEC, but also does so without relying on data augmentation or GEC-specific architecture changes. We further experiment with all possible configurations of our system with respect to model composition order and number of rounds of correction. A detailed analysis of each model and their contributions to the correction process is performed by adapting the ERRANT scorer to be able to score Chinese sentences.
    </details>
+ **Improving Grammatical Error Correction with Data Augmentation by Editing Latent Representation**
  + Authors: Zhaohong Wan, Xiaojun Wan, Wenguang Wang
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.200/
  + <details>
      <summary>Abstract</summary>
      The incorporation of data augmentation method in grammatical error correction task has attracted much attention. However, existing data augmentation methods mainly apply noise to tokens, which leads to the lack of diversity of generated errors. In view of this, we propose a new data augmentation method that can apply noise to the latent representation of a sentence. By editing the latent representations of grammatical sentences, we can generate synthetic samples with various error types. Combining with some pre-defined rules, our method can greatly improve the performance and robustness of existing grammatical error correction models. We evaluate our method on public benchmarks of GEC task and it achieves the state-of-the-art performance on CoNLL-2014 and FCE benchmarks.
    </details>
+ **SOME: Reference-less Sub-Metrics Optimized for Manual Evaluations of Grammatical Error Correction**
  + Authors: Ryoma Yoshimura, Masahiro Kaneko, Tomoyuki Kajiwara, Mamoru Komachi
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.573/
  + <details>
      <summary>Abstract</summary>
      We propose a reference-less metric trained on manual evaluations of system outputs for grammatical error correction (GEC). Previous studies have shown that reference-less metrics are promising; however, existing metrics are not optimized for manual evaluations of the system outputs because no dataset of the system output exists with manual evaluation. This study manually evaluates outputs of GEC systems to optimize the metrics. Experimental results show that the proposed metric improves correlation with the manual evaluation in both system- and sentence-level meta-evaluation. Our dataset and metric will be made publicly available.
    </details>
+ **Taking the Correction Difficulty into Account in Grammatical Error Correction Evaluation**
  + Authors: Takumi Gotou, Ryo Nagata, Masato Mita, Kazuaki Hanawa
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.188/
  + <details>
      <summary>Abstract</summary>
      This paper presents performance measures for grammatical error correction which take into account the difficulty of error correction. To the best of our knowledge, no conventional measure has such functionality despite the fact that some errors are easy to correct and others are not. The main purpose of this work is to provide a way of determining the difficulty of error correction and to motivate researchers in the domain to attack such difficult errors. The performance measures are based on the simple idea that the more systems successfully correct an error, the easier it is considered to be. This paper presents a set of algorithms to implement this idea. It evaluates the performance measures quantitatively and qualitatively on a wide variety of corpora and systems, revealing that they agree with our intuition of correction difficulty. A scorer and difficulty weight data based on the algorithms have been made available on the web.
    </details>
+ **MaskGEC: Improving Neural Grammatical Error Correction via Dynamic Masking**
  + Authors: Zewei Zhao, Houfeng Wang
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/5476

+ **Towards Minimal Supervision BERT-Based Grammar Error Correction (Student Abstract)**
  + Authors: Yiyuan Li, Antonios Anastasopoulos, Alan W. Black
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/7202

+ **Data Weighted Training Strategies for Grammatical Error Correction**
  + Authors: Jared Lichtarge, Chris Alberti, Shankar Kumar
  + Conference: TACL
  + Link: https://aclanthology.org/2020.tacl-1.41/
  + <details>
      <summary>Abstract</summary>
      Recent progress in the task of Grammatical Error Correction (GEC) has been driven by addressing data sparsity, both through new methods for generating large and noisy pretraining data and through the publication of small and higher-quality finetuning data in the BEA-2019 shared task. Building upon recent work in Neural Machine Translation (NMT), we make use of both kinds of data by deriving example-level scores on our large pretraining data based on a smaller, higher-quality dataset. In this work, we perform an empirical study to discover how to best incorporate delta-log-perplexity, a type of example scoring, into a training schedule for GEC. In doing so, we perform experiments that shed light on the function and applicability of delta-log-perplexity. Models trained on scored data achieve state- of-the-art results on common GEC test sets.
    </details>
+ **Classifying Syntactic Errors in Learner Language**
  + Authors: Leshem Choshen, Dmitry Nikolaev, Yevgeni Berzak, Omri Abend
  + Conference: CoNLL
  + Link: https://aclanthology.org/2020.conll-1.7/
  + <details>
      <summary>Abstract</summary>
      We present a method for classifying syntactic errors in learner language, namely errors whose correction alters the morphosyntactic structure of a sentence. The methodology builds on the established Universal Dependencies syntactic representation scheme, and provides complementary information to other error-classification systems. Unlike existing error classification methods, our method is applicable across languages, which we showcase by producing a detailed picture of syntactic errors in learner English and learner Russian. We further demonstrate the utility of the methodology for analyzing the outputs of leading Grammatical Error Correction (GEC) systems.
    </details>
+ **BERT Enhanced Neural Machine Translation and Sequence Tagging Model for Chinese Grammatical Error Diagnosis**
  + Authors: Deng Liang, Chen Zheng, Lei Guo, Xin Cui, Xiuzhang Xiong, Hengqiao Rong, Jinpeng Dong
  + Conference: AACL
  + Link: https://aclanthology.org/2020.nlptea-1.8/
  + <details>
      <summary>Abstract</summary>
      This paper presents the UNIPUS-Flaubert team’s hybrid system for the NLPTEA 2020 shared task of Chinese Grammatical Error Diagnosis (CGED). As a challenging NLP task, CGED has attracted increasing attention recently and has not yet fully benefited from the powerful pre-trained BERT-based models. We explore this by experimenting with three types of models. The position-tagging models and correction-tagging models are sequence tagging models fine-tuned on pre-trained BERT-based models, where the former focuses on detecting, positioning and classifying errors, and the latter aims at correcting errors. We also utilize rich representations from BERT-based models by transferring the BERT-fused models to the correction task, and further improve the performance by pre-training on a vast size of unsupervised synthetic data. To the best of our knowledge, we are the first to introduce and transfer the BERT-fused NMT model and sequence tagging model into the Chinese Grammatical Error Correction field. Our work achieved the second highest F1 score at the detecting errors, the best F1 score at correction top1 subtask and the second highest F1 score at correction top3 subtask.
    </details>
+ **Chinese Grammatical Correction Using BERT-based Pre-trained Model**
  + Authors: Hongfei Wang, Michiki Kurosawa, Satoru Katsumata, Mamoru Komachi
  + Conference: AACL
  + Link: https://aclanthology.org/2020.aacl-main.20/
  + <details>
      <summary>Abstract</summary>
      In recent years, pre-trained models have been extensively studied, and several downstream tasks have benefited from their utilization. In this study, we verify the effectiveness of two methods that incorporate a pre-trained model into an encoder-decoder model on Chinese grammatical error correction tasks. We also analyze the error type and conclude that sentence-level errors are yet to be addressed.
    </details>
+ **Chinese Grammatical Error Correction Based on Hybrid Models with Data Augmentation**
  + Authors: Yi Wang, Ruibin Yuan, Yan‘gen Luo, Yufang Qin, NianYong Zhu, Peng Cheng, Lihuan Wang
  + Conference: AACL
  + Link: https://aclanthology.org/2020.nlptea-1.10/
  + <details>
      <summary>Abstract</summary>
      A better Chinese Grammatical Error Diagnosis (CGED) system for automatic Grammatical Error Correction (GEC) can benefit foreign Chinese learners and lower Chinese learning barriers. In this paper, we introduce our solution to the CGED2020 Shared Task Grammatical Error Correction in detail. The task aims to detect and correct grammatical errors that occur in essays written by foreign Chinese learners. Our solution combined data augmentation methods, spelling check methods, and generative grammatical correction methods, and achieved the best recall score in the Top 1 Correction track. Our final result ranked fourth among the participants.
    </details>
+ **Chinese Grammatical Error Detection Based on BERT Model**
  + Authors: Yong Cheng, Mofan Duan
  + Conference: AACL
  + Link: https://aclanthology.org/2020.nlptea-1.15/
  + <details>
      <summary>Abstract</summary>
      Automatic grammatical error correction is of great value in assisting second language writing. In 2020, the shared task for Chinese grammatical error diagnosis(CGED) was held in NLP-TEA. As the LDU team, we participated the competition and submitted the final results. Our work mainly focused on grammatical error detection, that is, to judge whether a sentence contains grammatical errors. We used the BERT pre-trained model for binary classification, and we achieve 0.0391 in FPR track, ranking the second in all teams. In error detection track, the accuracy, recall and F-1 of our submitted result are 0.9851, 0.7496 and 0.8514 respectively.
    </details>
+ **Construction of an Evaluation Corpus for Grammatical Error Correction for Learners of Japanese as a Second Language**
  + Authors: Aomi Koyama, Tomoshige Kiyuna, Kenji Kobayashi, Mio Arai, Mamoru Komachi
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.26/
  + <details>
      <summary>Abstract</summary>
      The NAIST Lang-8 Learner Corpora (Lang-8 corpus) is one of the largest second-language learner corpora. The Lang-8 corpus is suitable as a training dataset for machine translation-based grammatical error correction systems. However, it is not suitable as an evaluation dataset because the corrected sentences sometimes include inappropriate sentences. Therefore, we created and released an evaluation corpus for correcting grammatical errors made by learners of Japanese as a Second Language (JSL). As our corpus has less noise and its annotation scheme reflects the characteristics of the dataset, it is ideal as an evaluation corpus for correcting grammatical errors in sentences written by JSL learners. In addition, we applied neural machine translation (NMT) and statistical machine translation (SMT) techniques to correct the grammar of the JSL learners’ sentences and evaluated their results using our corpus. We also compared the performance of the NMT system with that of the SMT system.
    </details>
+ **Developing NLP Tools with a New Corpus of Learner Spanish**
  + Authors: Sam Davidson, Aaron Yamada, Paloma Fernandez Mira, Agustina Carando, Claudia H. Sanchez Gutierrez, Kenji Sagae
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.894/
  + <details>
      <summary>Abstract</summary>
      The development of effective NLP tools for the L2 classroom depends largely on the availability of large annotated corpora of language learner text. While annotated learner corpora of English are widely available, large learner corpora of Spanish are less common. Those Spanish corpora that are available do not contain the annotations needed to facilitate the development of tools beneficial to language learners, such as grammatical error correction. As a result, the field has seen little research in NLP tools designed to benefit Spanish language learners and teachers. We introduce COWS-L2H, a freely available corpus of Spanish learner data which includes error annotations and parallel corrected text to help researchers better understand L2 development, to examine teaching practices empirically, and to develop NLP tools to better serve the Spanish teaching community. We demonstrate the utility of this corpus by developing a neural-network based grammatical error correction system for Spanish learner writing.
    </details>
+ **ERRANT: Assessing and Improving Grammatical Error Type Classification**
  + Authors: Katerina Korre, John Pavlopoulos
  + Conference: WS
  + Link: https://aclanthology.org/2020.latechclfl-1.10/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) is the task of correcting different types of errors in written texts. To manage this task, large amounts of annotated data that contain erroneous sentences are required. This data, however, is usually annotated according to each annotator’s standards, making it difficult to manage multiple sets of data at the same time. The recently introduced Error Annotation Toolkit (ERRANT) tackled this problem by presenting a way to automatically annotate data that contain grammatical errors, while also providing a standardisation for annotation. ERRANT extracts the errors and classifies them into error types, in the form of an edit that can be used in the creation of GEC systems, as well as for grammatical error analysis. However, we observe that certain errors are falsely or ambiguously classified. This could obstruct any qualitative or quantitative grammatical error type analysis, as the results would be inaccurate. In this work, we use a sample of the FCE coprus (Yannakoudakis et al., 2011) for secondary error type annotation and we show that up to 39% of the annotations of the most frequent type should be re-classified. Our corrections will be publicly released, so that they can serve as the starting point of a broader, collaborative, ongoing correction process.
    </details>
+ **GM-RKB WikiText Error Correction Task and Baselines**
  + Authors: Gabor Melli, Abdelrhman Eldallal, Bassim Lazem, Olga Moreira
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.295/
  + <details>
      <summary>Abstract</summary>
      We introduce the GM-RKB WikiText Error Correction Task for the automatic detection and correction of typographical errors in WikiText annotated pages. The included corpus is based on a snapshot of the GM-RKB domain-specific semantic wiki consisting of a large collection of concepts, personages, and publications primary centered on data mining and machine learning research topics. Numerous Wikipedia pages were also included as additional training data in the task’s evaluation process. The corpus was then automatically updated to synthetically include realistic errors to produce a training and evaluation ground truth comparison. We designed and evaluated two supervised baseline WikiFixer error correction methods: (1) a naive approach based on a maximum likelihood character-level language model; (2) and an advanced model based on a sequence-to-sequence (seq2seq) neural network architecture. Both error correction models operated at a character level. When compared against an off-the-shelf word-level spell checker these methods showed a significant improvement in the task’s performance – with the seq2seq-based model correcting a higher number of errors than it introduced. Finally, we published our data and code.
    </details>
+ **Gender-Aware Reinflection using Linguistically Enhanced Neural Models**
  + Authors: Bashar Alhafni, Nizar Habash, Houda Bouamor
  + Conference: WS
  + Link: https://aclanthology.org/2020.gebnlp-1.12/
  + <details>
      <summary>Abstract</summary>
      In this paper, we present an approach for sentence-level gender reinflection using linguistically enhanced sequence-to-sequence models. Our system takes an Arabic sentence and a given target gender as input and generates a gender-reinflected sentence based on the target gender. We formulate the problem as a user-aware grammatical error correction task and build an encoder-decoder architecture to jointly model reinflection for both masculine and feminine grammatical genders. We also show that adding linguistic features to our model leads to better reinflection results. The results on a blind test set using our best system show improvements over previous work, with a 3.6% absolute increase in M2 F0.5.
    </details>
+ **Generating Inflectional Errors for Grammatical Error Correction in Hindi**
  + Authors: Ankur Sonawane, Sujeet Kumar Vishwakarma, Bhavana Srivastava, Anil Kumar Singh
  + Conference: AACL
  + Link: https://aclanthology.org/2020.aacl-srw.24/
  + <details>
      <summary>Abstract</summary>
      Automated grammatical error correction has been explored as an important research problem within NLP, with the majority of the work being done on English and similar resource-rich languages. Grammar correction using neural networks is a data-heavy task, with the recent state of the art models requiring datasets with millions of annotated sentences for proper training. It is difficult to find such resources for Indic languages due to their relative lack of digitized content and complex morphology, compared to English. We address this problem by generating a large corpus of artificial inflectional errors for training GEC models. Moreover, to evaluate the performance of models trained on this dataset, we create a corpus of real Hindi errors extracted from Wikipedia edits. Analyzing this dataset with a modified version of the ERRANT error annotation toolkit, we find that inflectional errors are very common in this language. Finally, we produce the initial baseline results using state of the art methods developed for English.
    </details>
+ **GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors**
  + Authors: Masato Hagiwara, Masato Mita
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.835/
  + <details>
      <summary>Abstract</summary>
      The lack of large-scale datasets has been a major hindrance to the development of NLP tasks such as spelling correction and grammatical error correction (GEC). As a complementary new resource for these tasks, we present the GitHub Typo Corpus, a large-scale, multilingual dataset of misspellings and grammatical errors along with their corrections harvested from GitHub, a large and popular platform for hosting and sharing git repositories. The dataset, which we have made publicly available, contains more than 350k edits and 65M characters in more than 15 languages, making it the largest dataset of misspellings to date. We also describe our process for filtering true typo edits based on learned classifiers on a small annotated subset, and demonstrate that typo edits can be identified with F1 0.9 using a very simple classifier with only three features. The detailed analyses of the dataset show that existing spelling correctors merely achieve an F-measure of approx. 0.5, suggesting that the dataset serves as a new, rich source of spelling errors that complement existing datasets.
    </details>
+ **Non-Autoregressive Grammatical Error Correction Toward a Writing Support System**
  + Authors: Hiroki Homma, Mamoru Komachi
  + Conference: AACL
  + Link: https://aclanthology.org/2020.nlptea-1.1/
  + <details>
      <summary>Abstract</summary>
      There are several problems in applying grammatical error correction (GEC) to a writing support system. One of them is the handling of sentences in the middle of the input. Till date, the performance of GEC for incomplete sentences is not well-known. Hence, we analyze the performance of each model for incomplete sentences. Another problem is the correction speed. When the speed is slow, the usability of the system is limited, and the user experience is degraded. Therefore, in this study, we also focus on the non-autoregressive (NAR) model, which is a widely studied fast decoding method. We perform GEC in Japanese with traditional autoregressive and recent NAR models and analyze their accuracy and speed.
    </details>
+ **Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model**
  + Authors: Satoru Katsumata, Mamoru Komachi
  + Conference: AACL
  + Link: https://aclanthology.org/2020.aacl-main.83/
  + <details>
      <summary>Abstract</summary>
      Studies on grammatical error correction (GEC) have reported on the effectiveness of pretraining a Seq2Seq model with a large amount of pseudodata. However, this approach requires time-consuming pretraining of GEC because of the size of the pseudodata. In this study, we explored the utility of bidirectional and auto-regressive transformers (BART) as a generic pretrained encoder-decoder model for GEC. With the use of this generic pretrained model for GEC, the time-consuming pretraining can be eliminated. We find that monolingual and multilingual BART models achieve high performance in GEC, with one of the results being comparable to the current strong results in English GEC.
    </details>
+ **TMU-NLP System Using BERT-based Pre-trained Model to the NLP-TEA CGED Shared Task 2020**
  + Authors: Hongfei Wang, Mamoru Komachi
  + Conference: AACL
  + Link: https://aclanthology.org/2020.nlptea-1.11/
  + <details>
      <summary>Abstract</summary>
      In this paper, we introduce our system for NLPTEA 2020 shared task of Chinese Grammatical Error Diagnosis (CGED). In recent years, pre-trained models have been extensively studied, and several downstream tasks have benefited from their utilization. In this study, we treat the grammar error diagnosis (GED) task as a grammatical error correction (GEC) problem and propose a method that incorporates a pre-trained model into an encoder-decoder model to solve this problem.
    </details>
+ **面向汉语作为第二语言学习的个性化语法纠错(Personalizing Grammatical Error Correction for Chinese as a Second Language)**
  + Authors: Shengsheng Zhang (张生盛), Guina Pang (庞桂娜), Liner Yang (杨麟儿), Chencheng Wang (王辰成), Yongping Du (杜永萍), Erhong Yang (杨尔弘), Yaping Huang (黄雅平)
  + Conference: CCL
  + Link: https://aclanthology.org/2020.ccl-1.10/
  + <details>
      <summary>Abstract</summary>
      语法纠错任务旨在通过自然语言处理技术自动检测并纠正文本中的语序、拼写等语法错误。当前许多针对汉语的语法纠错方法已取得较好的效果,但往往忽略了学习者的个性化特征,如二语等级、母语背景等。因此,本文面向汉语作为第二语言的学习者,提出个性化语法纠错,对不同特征的学习者所犯的错误分别进行纠正,并构建了不同领域汉语学习者的数据集进行实验。实验结果表明,将语法纠错模型适应到学习者的各个领域后,性能得到明显提升。
    </details>
## 2019
+ **(Almost) Unsupervised Grammatical Error Correction using Synthetic Comparable Corpus**
  + Authors: Satoru Katsumata, Mamoru Komachi
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4413/
  + <details>
      <summary>Abstract</summary>
      We introduce unsupervised techniques based on phrase-based statistical machine translation for grammatical error correction (GEC) trained on a pseudo learner corpus created by Google Translation. We verified our GEC system through experiments on a low resource track of the shared task at BEA2019. As a result, we achieved an F0.5 score of 28.31 points with the test data.
    </details>
+ **A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning**
  + Authors: Yo Joong Choe, Jiyeon Ham, Kyubyong Park, Yeoil Yoon
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4423/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction can be viewed as a low-resource sequence-to-sequence task, because publicly available parallel corpora are limited. To tackle this challenge, we first generate erroneous versions of large unannotated corpora using a realistic noising function. The resulting parallel corpora are sub-sequently used to pre-train Transformer models. Then, by sequentially applying transfer learning, we adapt these models to the domain and style of the test set. Combined with a context-aware neural spellchecker, our system achieves competitive results in both restricted and low resource tracks in ACL 2019 BEAShared Task. We release all of our code and materials for reproducibility.
    </details>
+ **Artificial Error Generation with Fluency Filtering**
  + Authors: Mengyang Qiu, Jungyeul Park
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4408/
  + <details>
      <summary>Abstract</summary>
      The quantity and quality of training data plays a crucial role in grammatical error correction (GEC). However, due to the fact that obtaining human-annotated GEC data is both time-consuming and expensive, several studies have focused on generating artificial error sentences to boost training data for grammatical error correction, and shown significantly better performance. The present study explores how fluency filtering can affect the quality of artificial errors. By comparing artificial data filtered by different levels of fluency, we find that artificial error sentences with low fluency can greatly facilitate error correction, while high fluency errors introduce more noise.
    </details>
+ **Automatic Grammatical Error Correction for Sequence-to-sequence Text Generation: An Empirical Study**
  + Authors: Tao Ge, Xingxing Zhang, Furu Wei, Ming Zhou
  + Conference: ACL
  + Link: https://aclanthology.org/P19-1609/
  + <details>
      <summary>Abstract</summary>
      Sequence-to-sequence (seq2seq) models have achieved tremendous success in text generation tasks. However, there is no guarantee that they can always generate sentences without grammatical errors. In this paper, we present a preliminary empirical study on whether and how much automatic grammatical error correction can help improve seq2seq text generation. We conduct experiments across various seq2seq text generation tasks including machine translation, formality style transfer, sentence compression and simplification. Experiments show the state-of-the-art grammatical error correction system can improve the grammaticality of generated text and can bring task-oriented improvements in the tasks where target sentences are in a formal style.
    </details>
+ **CUNI System for the Building Educational Applications 2019 Shared Task: Grammatical Error Correction**
  + Authors: Jakub Náplava, Milan Straka
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4419/
  + <details>
      <summary>Abstract</summary>
      Our submitted models are NMT systems based on the Transformer model, which we improve by incorporating several enhancements: applying dropout to whole source and target words, weighting target subwords, averaging model checkpoints, and using the trained model iteratively for correcting the intermediate translations. The system in the Restricted Track is trained on the provided corpora with oversampled “cleaner” sentences and reaches 59.39 F0.5 score on the test set. The system in the Low-Resource Track is trained from Wikipedia revision histories and reaches 44.13 F0.5 score. Finally, we finetune the system from the Low-Resource Track on restricted data and achieve 64.55 F0.5 score.
    </details>
+ **Controlling Grammatical Error Correction Using Word Edit Rate**
  + Authors: Kengo Hotate, Masahiro Kaneko, Satoru Katsumata, Mamoru Komachi
  + Conference: ACL
  + Link: https://aclanthology.org/P19-2020/
  + <details>
      <summary>Abstract</summary>
      When professional English teachers correct grammatically erroneous sentences written by English learners, they use various methods. The correction method depends on how much corrections a learner requires. In this paper, we propose a method for neural grammar error correction (GEC) that can control the degree of correction. We show that it is possible to actually control the degree of GEC by using new training data annotated with word edit rate. Thereby, diverse corrected sentences is obtained from a single erroneous sentence. Moreover, compared to a GEC model that does not use information on the degree of correction, the proposed method improves correction accuracy.
    </details>
+ **Cross-Sentence Grammatical Error Correction**
  + Authors: Shamil Chollampatt, Weiqi Wang, Hwee Tou Ng
  + Conference: ACL
  + Link: https://aclanthology.org/P19-1042/
  + <details>
      <summary>Abstract</summary>
      Automatic grammatical error correction (GEC) research has made remarkable progress in the past decade. However, all existing approaches to GEC correct errors by considering a single sentence alone and ignoring crucial cross-sentence context. Some errors can only be corrected reliably using cross-sentence context and models can also benefit from the additional contextual information in correcting other errors. In this paper, we address this serious limitation of existing approaches and improve strong neural encoder-decoder models by appropriately modeling wider contexts. We employ an auxiliary encoder that encodes previous sentences and incorporate the encoding in the decoder via attention and gating mechanisms. Our approach results in statistically significant improvements in overall GEC performance over strong baselines across multiple test sets. Analysis of our cross-sentence GEC model on a synthetic dataset shows high performance in verb tense corrections that require cross-sentence context.
    </details>
+ **Erroneous data generation for Grammatical Error Correction**
  + Authors: Shuyao Xu, Jiehao Zhang, Jin Chen, Long Qin
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4415/
  + <details>
      <summary>Abstract</summary>
      It has been demonstrated that the utilization of a monolingual corpus in neural Grammatical Error Correction (GEC) systems can significantly improve the system performance. The previous state-of-the-art neural GEC system is an ensemble of four Transformer models pretrained on a large amount of Wikipedia Edits. The Singsound GEC system follows a similar approach but is equipped with a sophisticated erroneous data generating component. Our system achieved an F0:5 of 66.61 in the BEA 2019 Shared Task: Grammatical Error Correction. With our novel erroneous data generating component, the Singsound neural GEC system yielded an M2 of 63.2 on the CoNLL-2014 benchmark (8.4% relative improvement over the previous state-of-the-art system).
    </details>
+ **Improving Precision of Grammatical Error Correction with a Cheat Sheet**
  + Authors: Mengyang Qiu, Xuejiao Chen, Maggie Liu, Krishna Parvathala, Apurva Patil, Jungyeul Park
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4425/
  + <details>
      <summary>Abstract</summary>
      In this paper, we explore two approaches of generating error-focused phrases and examine whether these phrases can lead to better performance in grammatical error correction for the restricted track of BEA 2019 Shared Task on GEC. Our results show that phrases directly extracted from GEC corpora outperform phrases from statistical machine translation phrase table by a large margin. Appending error+context phrases to the original GEC corpora yields comparably high precision. We also explore the generation of artificial syntactic error sentences using error+context phrases for the unrestricted track. The additional training data greatly facilitates syntactic error correction (e.g., verb form) and contributes to better overall performance.
    </details>
+ **Learning to combine Grammatical Error Corrections**
  + Authors: Yoav Kantor, Yoav Katz, Leshem Choshen, Edo Cohen-Karlik, Naftali Liberman, Assaf Toledo, Amir Menczel, Noam Slonim
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4414/
  + <details>
      <summary>Abstract</summary>
      The field of Grammatical Error Correction (GEC) has produced various systems to deal with focused phenomena or general text editing. We propose an automatic way to combine black-box systems. Our method automatically detects the strength of a system or the combination of several systems per error type, improving precision and recall while optimizing F-score directly. We show consistent improvement over the best standalone system in all the configurations tested. This approach also outperforms average ensembling of different RNN models with random initializations. In addition, we analyze the use of BERT for GEC - reporting promising results on this end. We also present a spellchecker created for this task which outperforms standard spellcheckers tested on the task of spellchecking. This paper describes a system submission to Building Educational Applications 2019 Shared Task: Grammatical Error Correction. Combining the output of top BEA 2019 shared task systems using our approach, currently holds the highest reported score in the open phase of the BEA 2019 shared task, improving F-0.5 score by 3.7 points over the best result reported.
    </details>
+ **Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data**
  + Authors: Roman Grundkiewicz, Marcin Junczys-Dowmunt, Kenneth Heafield
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4427/
  + <details>
      <summary>Abstract</summary>
      Considerable effort has been made to address the data sparsity problem in neural grammatical error correction. In this work, we propose a simple and surprisingly effective unsupervised synthetic error generation method based on confusion sets extracted from a spellchecker to increase the amount of training data. Synthetic data is used to pre-train a Transformer sequence-to-sequence model, which not only improves over a strong baseline trained on authentic error-annotated data, but also enables the development of a practical GEC system in a scenario where little genuine error-annotated data is available. The developed systems placed first in the BEA19 shared task, achieving 69.47 and 64.24 F0.5 in the restricted and low-resource tracks respectively, both on the W&I+LOCNESS test set. On the popular CoNLL 2014 test set, we report state-of-the-art results of 64.16 M² for the submitted system, and 61.30 M² for the constrained system trained on the NUCLE and Lang-8 data.
    </details>
+ **Neural and FST-based approaches to grammatical error correction**
  + Authors: Zheng Yuan, Felix Stahlberg, Marek Rei, Bill Byrne, Helen Yannakoudakis
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4424/
  + <details>
      <summary>Abstract</summary>
      In this paper, we describe our submission to the BEA 2019 shared task on grammatical error correction. We present a system pipeline that utilises both error detection and correction models. The input text is first corrected by two complementary neural machine translation systems: one using convolutional networks and multi-task learning, and another using a neural Transformer-based system. Training is performed on publicly available data, along with artificial examples generated through back-translation. The n-best lists of these two machine translation systems are then combined and scored using a finite state transducer (FST). Finally, an unsupervised re-ranking system is applied to the n-best output of the FST. The re-ranker uses a number of error detection features to re-rank the FST n-best list and identify the final 1-best correction hypothesis. Our system achieves 66.75% F 0.5 on error correction (ranking 4th), and 82.52% F 0.5 on token-level error detection (ranking 2nd) in the restricted track of the shared task.
    </details>
+ **Noisy Channel for Low Resource Grammatical Error Correction**
  + Authors: Simon Flachs, Ophélie Lacroix, Anders Søgaard
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4420/
  + <details>
      <summary>Abstract</summary>
      This paper describes our contribution to the low-resource track of the BEA 2019 shared task on Grammatical Error Correction (GEC). Our approach to GEC builds on the theory of the noisy channel by combining a channel model and language model. We generate confusion sets from the Wikipedia edit history and use the frequencies of edits to estimate the channel model. Additionally, we use two pre-trained language models: 1) Google’s BERT model, which we fine-tune for specific error types and 2) OpenAI’s GPT-2 model, utilizing that it can operate with previous sentences as context. Furthermore, we search for the optimal combinations of corrections using beam search.
    </details>
+ **Online Infix Probability Computation for Probabilistic Finite Automata**
  + Authors: Marco Cognetta, Yo-Sub Han, Soon Chan Kwon
  + Conference: ACL
  + Link: https://aclanthology.org/P19-1528/
  + <details>
      <summary>Abstract</summary>
      Probabilistic finite automata (PFAs) are com- mon statistical language model in natural lan- guage and speech processing. A typical task for PFAs is to compute the probability of all strings that match a query pattern. An impor- tant special case of this problem is computing the probability of a string appearing as a pre- fix, suffix, or infix. These problems find use in many natural language processing tasks such word prediction and text error correction. Recently, we gave the first incremental algorithm to efficiently compute the infix probabilities of each prefix of a string (Cognetta et al., 2018). We develop an asymptotic improvement of that algorithm and solve the open problem of computing the infix probabilities of PFAs from streaming data, which is crucial when process- ing queries online and is the ultimate goal of the incremental approach.
    </details>
+ **TMU Transformer System Using BERT for Re-ranking at BEA 2019 Grammatical Error Correction on Restricted Track**
  + Authors: Masahiro Kaneko, Kengo Hotate, Satoru Katsumata, Mamoru Komachi
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4422/
  + <details>
      <summary>Abstract</summary>
      We introduce our system that is submitted to the restricted track of the BEA 2019 shared task on grammatical error correction1 (GEC). It is essential to select an appropriate hypothesis sentence from the candidates list generated by the GEC model. A re-ranker can evaluate the naturalness of a corrected sentence using language models trained on large corpora. On the other hand, these language models and language representations do not explicitly take into account the grammatical errors written by learners. Thus, it is not straightforward to utilize language representations trained from a large corpus, such as Bidirectional Encoder Representations from Transformers (BERT), in a form suitable for the learner’s grammatical errors. Therefore, we propose to fine-tune BERT on learner corpora with grammatical errors for re-ranking. The experimental results of the W&I+LOCNESS development dataset demonstrate that re-ranking using BERT can effectively improve the correction performance.
    </details>
+ **The AIP-Tohoku System at the BEA-2019 Shared Task**
  + Authors: Hiroki Asano, Masato Mita, Tomoya Mizumoto, Jun Suzuki
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4418/
  + <details>
      <summary>Abstract</summary>
      We introduce the AIP-Tohoku grammatical error correction (GEC) system for the BEA-2019 shared task in Track 1 (Restricted Track) and Track 2 (Unrestricted Track) using the same system architecture. Our system comprises two key components: error generation and sentence-level error detection. In particular, GEC with sentence-level grammatical error detection is a novel and versatile approach, and we experimentally demonstrate that it significantly improves the precision of the base model. Our system is ranked 9th in Track 1 and 2nd in Track 2.
    </details>
+ **The BEA-2019 Shared Task on Grammatical Error Correction**
  + Authors: Christopher Bryant, Mariano Felice, Øistein E. Andersen, Ted Briscoe
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4406/
  + <details>
      <summary>Abstract</summary>
      This paper reports on the BEA-2019 Shared Task on Grammatical Error Correction (GEC). As with the CoNLL-2014 shared task, participants are required to correct all types of errors in test data. One of the main contributions of the BEA-2019 shared task is the introduction of a new dataset, the Write&Improve+LOCNESS corpus, which represents a wider range of native and learner English levels and abilities. Another contribution is the introduction of tracks, which control the amount of annotated data available to participants. Systems are evaluated in terms of ERRANT F_0.5, which allows us to report a much wider range of performance statistics. The competition was hosted on Codalab and remains open for further submissions on the blind test set.
    </details>
+ **The BLCU System in the BEA 2019 Shared Task**
  + Authors: Liner Yang, Chencheng Wang
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4421/
  + <details>
      <summary>Abstract</summary>
      This paper describes the BLCU Group submissions to the Building Educational Applications (BEA) 2019 Shared Task on Grammatical Error Correction (GEC). The task is to detect and correct grammatical errors that occurred in essays. We participate in 2 tracks including the Restricted Track and the Unrestricted Track. Our system is based on a Transformer model architecture. We integrate many effective methods proposed in recent years. Such as, Byte Pair Encoding, model ensemble, checkpoints average and spell checker. We also corrupt the public monolingual data to further improve the performance of the model. On the test data of the BEA 2019 Shared Task, our system yields F0.5 = 58.62 and 59.50, ranking twelfth and fourth respectively.
    </details>
+ **The CUED’s Grammatical Error Correction Systems for BEA-2019**
  + Authors: Felix Stahlberg, Bill Byrne
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4417/
  + <details>
      <summary>Abstract</summary>
      We describe two entries from the Cambridge University Engineering Department to the BEA 2019 Shared Task on grammatical error correction. Our submission to the low-resource track is based on prior work on using finite state transducers together with strong neural language models. Our system for the restricted track is a purely neural system consisting of neural language models and neural machine translation models trained with back-translation and a combination of checkpoint averaging and fine-tuning – without the help of any additional tools like spell checkers. The latter system has been used inside a separate system combination entry in cooperation with the Cambridge University Computer Lab.
    </details>
+ **The Unbearable Weight of Generating Artificial Errors for Grammatical Error Correction**
  + Authors: Phu Mon Htut, Joel Tetreault
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4449/
  + <details>
      <summary>Abstract</summary>
      In this paper, we investigate the impact of using 4 recent neural models for generating artificial errors to help train the neural grammatical error correction models. We conduct a battery of experiments on the effect of data size, models, and comparison with a rule-based approach.
    </details>
+ **The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction**
  + Authors: Dimitris Alikaniotis, Vipul Raheja
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4412/
  + <details>
      <summary>Abstract</summary>
      Recent work on Grammatical Error Correction (GEC) has highlighted the importance of language modeling in that it is certainly possible to achieve good performance by comparing the probabilities of the proposed edits. At the same time, advancements in language modeling have managed to generate linguistic output, which is almost indistinguishable from that of human-generated text. In this paper, we up the ante by exploring the potential of more sophisticated language models in GEC and offer some key insights on their strengths and weaknesses. We show that, in line with recent results in other NLP tasks, Transformer architectures achieve consistently high performance and provide a competitive baseline for future machine learning models.
    </details>
+ **An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction**
  + Authors: Shun Kiyono, Jun Suzuki, Masato Mita, Tomoya Mizumoto, Kentaro Inui
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-1119/
  + <details>
      <summary>Abstract</summary>
      The incorporation of pseudo data in the training of grammatical error correction models has been one of the main factors in improving the performance of such models. However, consensus is lacking on experimental configurations, namely, choosing how the pseudo data should be generated or used. In this study, these choices are investigated through extensive experiments, and state-of-the-art performance is achieved on the CoNLL-2014 test set (F0.5=65.0) and the official test set of the BEA-2019 shared task (F0.5=70.2) without making any modifications to the model architecture.
    </details>
+ **Denoising based Sequence-to-Sequence Pre-training for Text Generation**
  + Authors: Liang Wang, Wei Zhao, Ruoyu Jia, Sujian Li, Jingming Liu
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-1412/
  + <details>
      <summary>Abstract</summary>
      This paper presents a new sequence-to-sequence (seq2seq) pre-training method PoDA (Pre-training of Denoising Autoencoders), which learns representations suitable for text generation tasks. Unlike encoder-only (e.g., BERT) or decoder-only (e.g., OpenAI GPT) pre-training approaches, PoDA jointly pre-trains both the encoder and decoder by denoising the noise-corrupted text, and it also has the advantage of keeping the network architecture unchanged in the subsequent fine-tuning stage. Meanwhile, we design a hybrid model of Transformer and pointer-generator networks as the backbone architecture for PoDA. We conduct experiments on two text generation tasks: abstractive summarization, and grammatical error correction. Results on four datasets show that PoDA can improve model performance over strong baselines without using any task-specific techniques and significantly speed up convergence.
    </details>
+ **Grammatical Error Correction in Low-Resource Scenarios**
  + Authors: Jakub Náplava, Milan Straka
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-5545/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction in English is a long studied problem with many existing systems and datasets. However, there has been only a limited research on error correction of other languages. In this paper, we present a new dataset AKCES-GEC on grammatical error correction for Czech. We then make experiments on Czech, German and Russian and show that when utilizing synthetic parallel corpus, Transformer neural machine translation model can reach new state-of-the-art results on these datasets. AKCES-GEC is published under CC BY-NC-SA 4.0 license at http://hdl.handle.net/11234/1-3057, and the source code of the GEC model is available at https://github.com/ufal/low-resource-gec-wnut2019.
    </details>
+ **Minimally-Augmented Grammatical Error Correction**
  + Authors: Roman Grundkiewicz, Marcin Junczys-Dowmunt
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-5546/
  + <details>
      <summary>Abstract</summary>
      There has been an increased interest in low-resource approaches to automatic grammatical error correction. We introduce Minimally-Augmented Grammatical Error Correction (MAGEC) that does not require any error-labelled data. Our unsupervised approach is based on a simple but effective synthetic error generation method based on confusion sets from inverted spell-checkers. In low-resource settings, we outperform the current state-of-the-art results for German and Russian GEC tasks by a large margin without using any real error-annotated training data. When combined with labelled data, our method can serve as an efficient pre-training technique
    </details>
+ **Parallel Iterative Edit Models for Local Sequence Transduction**
  + Authors: Abhijeet Awasthi, Sunita Sarawagi, Rasna Goyal, Sabyasachi Ghosh, Vihari Piratla
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-1435/
  + <details>
      <summary>Abstract</summary>
      We present a Parallel Iterative Edit (PIE) model for the problem of local sequence transduction arising in tasks like Grammatical error correction (GEC). Recent approaches are based on the popular encoder-decoder (ED) model for sequence to sequence learning. The ED model auto-regressively captures full dependency among output tokens but is slow due to sequential decoding. The PIE model does parallel decoding, giving up the advantage of modeling full dependency in the output, yet it achieves accuracy competitive with the ED model for four reasons: 1. predicting edits instead of tokens, 2. labeling sequences instead of generating sequences, 3. iteratively refining predictions to capture dependencies, and 4. factorizing logits over edits and their token argument to harness pre-trained language models like BERT. Experiments on tasks spanning GEC, OCR correction and spell correction demonstrate that the PIE model is an accurate and significantly faster alternative for local sequence transduction.
    </details>
+ **Personalizing Grammatical Error Correction: Adaptation to Proficiency Level and L1**
  + Authors: Maria Nadejde, Joel Tetreault
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-5504/
  + <details>
      <summary>Abstract</summary>
      Grammar error correction (GEC) systems have become ubiquitous in a variety of software applications, and have started to approach human-level performance for some datasets. However, very little is known about how to efficiently personalize these systems to the user’s characteristics, such as their proficiency level and first language, or to emerging domains of text. We present the first results on adapting a general purpose neural GEC system to both the proficiency level and the first language of a writer, using only a few thousand annotated sentences. Our study is the broadest of its kind, covering five proficiency levels and twelve different languages, and comparing three different adaptation scenarios: adapting to the proficiency level only, to the first language only, or to both aspects simultaneously. We show that tailoring to both scenarios achieves the largest performance improvement (3.6 F0.5) relative to a strong baseline.
    </details>
+ **TEASPN: Framework and Protocol for Integrated Writing Assistance Environments**
  + Authors: Masato Hagiwara, Takumi Ito, Tatsuki Kuribayashi, Jun Suzuki, Kentaro Inui
  + Conference: EMNLP
  + Link: https://aclanthology.org/D19-3039/
  + <details>
      <summary>Abstract</summary>
      Language technologies play a key role in assisting people with their writing. Although there has been steady progress in e.g., grammatical error correction (GEC), human writers are yet to benefit from this progress due to the high development cost of integrating with writing software. We propose TEASPN, a protocol and an open-source framework for achieving integrated writing assistance environments. The protocol standardizes the way writing software communicates with servers that implement such technologies, allowing developers and researchers to integrate the latest developments in natural language processing (NLP) with low cost. As a result, users can enjoy the integrated experience in their favorite writing software. The results from experiments with human participants show that users use a wide range of technologies and rate their writing experience favorably, allowing them to write more fluent text.
    </details>
+ **Corpora Generation for Grammatical Error Correction**
  + Authors: Jared Lichtarge, Chris Alberti, Shankar Kumar, Noam Shazeer, Niki Parmar, Simon Tong
  + Conference: NAACL
  + Link: https://aclanthology.org/N19-1333/
  + <details>
      <summary>Abstract</summary>
      Grammatical Error Correction (GEC) has been recently modeled using the sequence-to-sequence framework. However, unlike sequence transduction problems such as machine translation, GEC suffers from the lack of plentiful parallel data. We describe two approaches for generating large parallel datasets for GEC using publicly available Wikipedia data. The first method extracts source-target pairs from Wikipedia edit histories with minimal filtration heuristics while the second method introduces noise into Wikipedia sentences via round-trip translation through bridge languages. Both strategies yield similar sized parallel corpora containing around 4B tokens. We employ an iterative decoding strategy that is tailored to the loosely supervised nature of our constructed corpora. We demonstrate that neural GEC models trained using either type of corpora give similar performance. Fine-tuning these models on the Lang-8 corpus and ensembling allows us to surpass the state of the art on both the CoNLL ‘14 benchmark and the JFLEG task. We present systematic analysis that compares the two approaches to data generation and highlights the effectiveness of ensembling.
    </details>
+ **Cross-Corpora Evaluation and Analysis of Grammatical Error Correction Models — Is Single-Corpus Evaluation Enough?**
  + Authors: Masato Mita, Tomoya Mizumoto, Masahiro Kaneko, Ryo Nagata, Kentaro Inui
  + Conference: NAACL
  + Link: https://aclanthology.org/N19-1132/
  + <details>
      <summary>Abstract</summary>
      This study explores the necessity of performing cross-corpora evaluation for grammatical error correction (GEC) models. GEC models have been previously evaluated based on a single commonly applied corpus: the CoNLL-2014 benchmark. However, the evaluation remains incomplete because the task difficulty varies depending on the test corpus and conditions such as the proficiency levels of the writers and essay topics. To overcome this limitation, we evaluate the performance of several GEC models, including NMT-based (LSTM, CNN, and transformer) and an SMT-based model, against various learner corpora (CoNLL-2013, CoNLL-2014, FCE, JFLEG, ICNALE, and KJ). Evaluation results reveal that the models’ rankings considerably vary depending on the corpus, indicating that single-corpus evaluation is insufficient for GEC models.
    </details>
+ **Generate, Filter, and Rank: Grammaticality Classification for Production-Ready NLG Systems**
  + Authors: Ashwini Challa, Kartikeya Upasani, Anusha Balakrishnan, Rajen Subba
  + Conference: NAACL
  + Link: https://aclanthology.org/N19-2027/
  + <details>
      <summary>Abstract</summary>
      Neural approaches to Natural Language Generation (NLG) have been promising for goal-oriented dialogue. One of the challenges of productionizing these approaches, however, is the ability to control response quality, and ensure that generated responses are acceptable. We propose the use of a generate, filter, and rank framework, in which candidate responses are first filtered to eliminate unacceptable responses, and then ranked to select the best response. While acceptability includes grammatical correctness and semantic correctness, we focus only on grammaticality classification in this paper, and show that existing datasets for grammatical error correction don’t correctly capture the distribution of errors that data-driven generators are likely to make. We release a grammatical classification and semantic correctness classification dataset for the weather domain that consists of responses generated by 3 data-driven NLG systems. We then explore two supervised learning approaches (CNNs and GBDTs) for classifying grammaticality. Our experiments show that grammaticality classification is very sensitive to the distribution of errors in the data, and that these distributions vary significantly with both the source of the response as well as the domain. We show that it’s possible to achieve high precision with reasonable recall on our dataset.
    </details>
+ **Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data**
  + Authors: Wei Zhao, Liang Wang, Kewei Shen, Ruoyu Jia, Jingming Liu
  + Conference: NAACL
  + Link: https://aclanthology.org/N19-1014/
  + <details>
      <summary>Abstract</summary>
      Neural machine translation systems have become state-of-the-art approaches for Grammatical Error Correction (GEC) task. In this paper, we propose a copy-augmented architecture for the GEC task by copying the unchanged words from the source sentence to the target sentence. Since the GEC suffers from not having enough labeled training data to achieve high accuracy. We pre-train the copy-augmented architecture with a denoising auto-encoder using the unlabeled One Billion Benchmark and make comparisons between the fully pre-trained model and a partially pre-trained model. It is the first time copying words from the source context and fully pre-training a sequence to sequence model are experimented on the GEC task. Moreover, We add token-level and sentence-level multi-task learning for the GEC task. The evaluation results on the CoNLL-2014 test set show that our approach outperforms all recently published state-of-the-art results by a large margin.
    </details>
+ **Neural Grammatical Error Correction with Finite State Transducers**
  + Authors: Felix Stahlberg, Christopher Bryant, Bill Byrne
  + Conference: NAACL
  + Link: https://aclanthology.org/N19-1406/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) is one of the areas in natural language processing in which purely neural models have not yet superseded more traditional symbolic models. Hybrid systems combining phrase-based statistical machine translation (SMT) and neural sequence models are currently among the most effective approaches to GEC. However, both SMT and neural sequence-to-sequence models require large amounts of annotated data. Language model based GEC (LM-GEC) is a promising alternative which does not rely on annotated training data. We show how to improve LM-GEC by applying modelling techniques based on finite state transducers. We report further gains by rescoring with neural language models. We show that our methods developed for LM-GEC can also be used with SMT systems if annotated training data is available. Our best system outperforms the best published result on the CoNLL-2014 test set, and achieves far better relative improvements over the SMT baselines than previous hybrid systems.
    </details>
+ **Neural Machine Translation of Text from Non-Native Speakers**
  + Authors: Antonios Anastasopoulos, Alison Lui, Toan Q. Nguyen, David Chiang
  + Conference: NAACL
  + Link: https://aclanthology.org/N19-1311/
  + <details>
      <summary>Abstract</summary>
      Neural Machine Translation (NMT) systems are known to degrade when confronted with noisy data, especially when the system is trained only on clean data. In this paper, we show that augmenting training data with sentences containing artificially-introduced grammatical errors can make the system more robust to such errors. In combination with an automatic grammar error correction system, we can recover 1.0 BLEU out of 2.4 BLEU lost due to grammatical errors. We also present a set of Spanish translations of the JFLEG grammar error correction corpus, which allows for testing NMT robustness to real grammatical errors.
    </details>
+ **Enabling Robust Grammatical Error Correction in New Domains: Data Sets, Metrics, and Analyses**
  + Authors: Courtney Napoles, Maria Nădejde, Joel Tetreault
  + Conference: TACL
  + Link: https://aclanthology.org/Q19-1032/
  + <details>
      <summary>Abstract</summary>
      Until now, grammatical error correction (GEC) has been primarily evaluated on text written by non-native English speakers, with a focus on student essays. This paper enables GEC development on text written by native speakers by providing a new data set and metric. We present a multiple-reference test corpus for GEC that includes 4,000 sentences in two new domains (formal and informal writing by native English speakers) and 2,000 sentences from a diverse set of non-native student writing. We also collect human judgments of several GEC systems on this new test set and perform a meta-evaluation, assessing how reliable automatic metrics are across these domains. We find that commonly used GEC metrics have inconsistent performance across domains, and therefore we propose a new ensemble metric that is robust on all three domains of text.
    </details>
+ **Grammar Error Correction in Morphologically Rich Languages: The Case of Russian**
  + Authors: Alla Rozovskaya, Dan Roth
  + Conference: TACL
  + Link: https://aclanthology.org/Q19-1001/
  + <details>
      <summary>Abstract</summary>
      Until now, most of the research in grammar error correction focused on English, and the problem has hardly been explored for other languages. We address the task of correcting writing mistakes in morphologically rich languages, with a focus on Russian. We present a corrected and error-tagged corpus of Russian learner writing and develop models that make use of existing state-of-the-art methods that have been well studied for English. Although impressive results have recently been achieved for grammar error correction of non-native English writing, these results are limited to domains where plentiful training data are available. Because annotation is extremely costly, these approaches are not suitable for the majority of domains and languages. We thus focus on methods that use “minimal supervision”; that is, those that do not rely on large amounts of annotated training data, and show how existing minimal-supervision approaches extend to a highly inflectional language such as Russian. The results demonstrate that these methods are particularly useful for correcting mistakes in grammatical phenomena that involve rich morphology.
    </details>
+ **Diamonds in the Rough: Generating Fluent Sentences from Early-Stage Drafts for Academic Writing Assistance**
  + Authors: Takumi Ito, Tatsuki Kuribayashi, Hayato Kobayashi, Ana Brassard, Masato Hagiwara, Jun Suzuki, Kentaro Inui
  + Conference: WS
  + Link: https://aclanthology.org/W19-8606/
  + <details>
      <summary>Abstract</summary>
      The writing process consists of several stages such as drafting, revising, editing, and proofreading. Studies on writing assistance, such as grammatical error correction (GEC), have mainly focused on sentence editing and proofreading, where surface-level issues such as typographical errors, spelling errors, or grammatical errors should be corrected. We broaden this focus to include the earlier revising stage, where sentences require adjustment to the information included or major rewriting and propose Sentence-level Revision (SentRev) as a new writing assistance task. Well-performing systems in this task can help inexperienced authors by producing fluent, complete sentences given their rough, incomplete drafts. We build a new freely available crowdsourced evaluation dataset consisting of incomplete sentences authored by non-native writers paired with their final versions extracted from published academic papers for developing and evaluating SentRev models. We also establish baseline performance on SentRev using our newly built evaluation dataset.
    </details>
## 2018
+ **A Hybrid System for Chinese Grammatical Error Diagnosis and Correction**
  + Authors: Chen Li, Junpei Zhou, Zuyi Bao, Hengyou Liu, Guangwei Xu, Linlin Li
  + Conference: ACL
  + Link: https://aclanthology.org/W18-3708/
  + <details>
      <summary>Abstract</summary>
      This paper introduces the DM_NLP team’s system for NLPTEA 2018 shared task of Chinese Grammatical Error Diagnosis (CGED), which can be used to detect and correct grammatical errors in texts written by Chinese as a Foreign Language (CFL) learners. This task aims at not only detecting four types of grammatical errors including redundant words (R), missing words (M), bad word selection (S) and disordered words (W), but also recommending corrections for errors of M and S types. We proposed a hybrid system including four models for this task with two stages: the detection stage and the correction stage. In the detection stage, we first used a BiLSTM-CRF model to tag potential errors by sequence labeling, along with some handcraft features. Then we designed three Grammatical Error Correction (GEC) models to generate corrections, which could help to tune the detection result. In the correction stage, candidates were generated by the three GEC models and then merged to output the final corrections for M and S types. Our system reached the highest precision in the correction subtask, which was the most challenging part of this shared task, and got top 3 on F1 scores for position detection of errors.
    </details>
+ **Automatic Metric Validation for Grammatical Error Correction**
  + Authors: Leshem Choshen, Omri Abend
  + Conference: ACL
  + Link: https://aclanthology.org/P18-1127/
  + <details>
      <summary>Abstract</summary>
      Metric validation in Grammatical Error Correction (GEC) is currently done by observing the correlation between human and metric-induced rankings. However, such correlation studies are costly, methodologically troublesome, and suffer from low inter-rater agreement. We propose MAEGE, an automatic methodology for GEC metric validation, that overcomes many of the difficulties in the existing methodology. Experiments with MAEGE shed a new light on metric quality, showing for example that the standard M2 metric fares poorly on corpus-level ranking. Moreover, we use MAEGE to perform a detailed analysis of metric behavior, showing that some types of valid edits are consistently penalized by existing metrics.
    </details>
+ **Fluency Boost Learning and Inference for Neural Grammatical Error Correction**
  + Authors: Tao Ge, Furu Wei, Ming Zhou
  + Conference: ACL
  + Link: https://aclanthology.org/P18-1097/
  + <details>
      <summary>Abstract</summary>
      Most of the neural sequence-to-sequence (seq2seq) models for grammatical error correction (GEC) have two limitations: (1) a seq2seq model may not be well generalized with only limited error-corrected data; (2) a seq2seq model may fail to completely correct a sentence with multiple errors through normal seq2seq inference. We attempt to address these limitations by proposing a fluency boost learning and inference mechanism. Fluency boosting learning generates fluency-boost sentence pairs during training, enabling the error correction model to learn how to improve a sentence’s fluency from more instances, while fluency boosting inference allows the model to correct a sentence incrementally with multiple inference steps until the sentence’s fluency stops increasing. Experiments show our approaches improve the performance of seq2seq models for GEC, achieving state-of-the-art results on both CoNLL-2014 and JFLEG benchmark datasets.
    </details>
+ **Graph-based Filtering of Out-of-Vocabulary Words for Encoder-Decoder Models**
  + Authors: Satoru Katsumata, Yukio Matsumura, Hayahide Yamagishi, Mamoru Komachi
  + Conference: ACL
  + Link: https://aclanthology.org/P18-3016/
  + <details>
      <summary>Abstract</summary>
      Encoder-decoder models typically only employ words that are frequently used in the training corpus because of the computational costs and/or to exclude noisy words. However, this vocabulary set may still include words that interfere with learning in encoder-decoder models. This paper proposes a method for selecting more suitable words for learning encoders by utilizing not only frequency, but also co-occurrence information, which we capture using the HITS algorithm. The proposed method is applied to two tasks: machine translation and grammatical error correction. For Japanese-to-English translation, this method achieved a BLEU score that was 0.56 points more than that of a baseline. It also outperformed the baseline method for English grammatical error correction, with an F-measure that was 1.48 points higher.
    </details>
+ **Inherent Biases in Reference-based Evaluation for Grammatical Error Correction**
  + Authors: Leshem Choshen, Omri Abend
  + Conference: ACL
  + Link: https://aclanthology.org/P18-1059/
  + <details>
      <summary>Abstract</summary>
      The prevalent use of too few references for evaluating text-to-text generation is known to bias estimates of their quality (henceforth, low coverage bias or LCB). This paper shows that overcoming LCB in Grammatical Error Correction (GEC) evaluation cannot be attained by re-scaling or by increasing the number of references in any feasible range, contrary to previous suggestions. This is due to the long-tailed distribution of valid corrections for a sentence. Concretely, we show that LCB incentivizes GEC systems to avoid correcting even when they can generate a valid correction. Consequently, existing systems obtain comparable or superior performance compared to humans, by making few but targeted changes to the input. Similar effects on Text Simplification further support our claims.
    </details>
+ **Ling@CASS Solution to the NLP-TEA CGED Shared Task 2018**
  + Authors: Qinan Hu, Yongwei Zhang, Fang Liu, Yueguo Gu
  + Conference: ACL
  + Link: https://aclanthology.org/W18-3709/
  + <details>
      <summary>Abstract</summary>
      In this study, we employ the sequence to sequence learning to model the task of grammar error correction. The system takes potentially erroneous sentences as inputs, and outputs correct sentences. To breakthrough the bottlenecks of very limited size of manually labeled data, we adopt a semi-supervised approach. Specifically, we adapt correct sentences written by native Chinese speakers to generate pseudo grammatical errors made by learners of Chinese as a second language. We use the pseudo data to pre-train the model, and the CGED data to fine-tune it. Being aware of the significance of precision in a grammar error correction system in real scenarios, we use ensembles to boost precision. When using inputs as simple as Chinese characters, the ensembled system achieves a precision at 86.56% in the detection of erroneous sentences, and a precision at 51.53% in the correction of errors of Selection and Missing types.
    </details>
+ **How do you correct run-on sentences it’s not as easy as it seems**
  + Authors: Junchao Zheng, Courtney Napoles, Joel Tetreault, Kostiantyn Omelianchuk
  + Conference: EMNLP
  + Link: https://aclanthology.org/W18-6105/
  + <details>
      <summary>Abstract</summary>
      Run-on sentences are common grammatical mistakes but little research has tackled this problem to date. This work introduces two machine learning models to correct run-on sentences that outperform leading methods for related tasks, punctuation restoration and whole-sentence grammatical error correction. Due to the limited annotated data for this error, we experiment with artificially generating training data from clean newswire text. Our findings suggest artificial training data is viable for this task. We discuss implications for correcting run-ons and other types of mistakes that have low coverage in error-annotated corpora.
    </details>
+ **Neural Quality Estimation of Grammatical Error Correction**
  + Authors: Shamil Chollampatt, Hwee Tou Ng
  + Conference: EMNLP
  + Link: https://aclanthology.org/D18-1274/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction (GEC) systems deployed in language learning environments are expected to accurately correct errors in learners’ writing. However, in practice, they often produce spurious corrections and fail to correct many errors, thereby misleading learners. This necessitates the estimation of the quality of output sentences produced by GEC systems so that instructors can selectively intervene and re-correct the sentences which are poorly corrected by the system and ensure that learners get accurate feedback. We propose the first neural approach to automatic quality estimation of GEC output sentences that does not employ any hand-crafted features. Our system is trained in a supervised manner on learner sentences and corresponding GEC system outputs with quality score labels computed using human-annotated references. Our neural quality estimation models for GEC show significant improvements over a strong feature-based baseline. We also show that a state-of-the-art GEC system can be improved when quality scores are used as features for re-ranking the N-best candidates.
    </details>
+ **Using Wikipedia Edits in Low Resource Grammatical Error Correction**
  + Authors: Adriane Boyd
  + Conference: EMNLP
  + Link: https://aclanthology.org/W18-6111/
  + <details>
      <summary>Abstract</summary>
      We develop a grammatical error correction (GEC) system for German using a small gold GEC corpus augmented with edits extracted from Wikipedia revision history. We extend the automatic error annotation tool ERRANT (Bryant et al., 2017) for German and use it to analyze both gold GEC corrections and Wikipedia edits (Grundkiewicz and Junczys-Dowmunt, 2014) in order to select as additional training data Wikipedia edits containing grammatical corrections similar to those in the gold corpus. Using a multilayer convolutional encoder-decoder neural network GEC approach (Chollampatt and Ng, 2018), we evaluate the contribution of Wikipedia edits and find that carefully selected Wikipedia edits increase performance by over 5%.
    </details>
+ **Wronging a Right: Generating Better Errors to Improve Grammatical Error Detection**
  + Authors: Sudhanshu Kasewa, Pontus Stenetorp, Sebastian Riedel
  + Conference: EMNLP
  + Link: https://aclanthology.org/D18-1541/
  + <details>
      <summary>Abstract</summary>
      Grammatical error correction, like other machine learning tasks, greatly benefits from large quantities of high quality training data, which is typically expensive to produce. While writing a program to automatically generate realistic grammatical errors would be difficult, one could learn the distribution of naturally-occurring errors and attempt to introduce them into other datasets. Initial work on inducing errors in this way using statistical machine translation has shown promise; we investigate cheaply constructing synthetic samples, given a small corpus of human-annotated data, using an off-the-rack attentive sequence-to-sequence model and a straight-forward post-processing procedure. Our approach yields error-filled artificial data that helps a vanilla bi-directional LSTM to outperform the previous state of the art at grammatical error detection, and a previously introduced model to gain further improvements of over 5% F0.5 score. When attempting to determine if a given sentence is synthetic, a human annotator at best achieves 39.39 F1 score, indicating that our model generates mostly human-like instances.
    </details>
+ **Approaching Neural Grammatical Error Correction as a Low-Resource Machine Translation Task**
  + Authors: Marcin Junczys-Dowmunt, Roman Grundkiewicz, Shubha Guha, Kenneth Heafield
  + Conference: NAACL
  + Link: https://aclanthology.org/N18-1055/
  + <details>
      <summary>Abstract</summary>
      Previously, neural methods in grammatical error correction (GEC) did not reach state-of-the-art results compared to phrase-based statistical machine translation (SMT) baselines. We demonstrate parallels between neural GEC and low-resource neural MT and successfully adapt several methods from low-resource MT to neural GEC. We further establish guidelines for trustable results in neural GEC and propose a set of model-independent methods for neural GEC that can be easily applied in most GEC settings. Proposed methods include adding source-side noise, domain-adaptation techniques, a GEC-specific training-objective, transfer learning with monolingual data, and ensembling of independently trained GEC models and language models. The combined effects of these methods result in better than state-of-the-art neural GEC models that outperform previously best neural GEC systems by more than 10% M² on the CoNLL-2014 benchmark and 5.9% on the JFLEG test set. Non-neural state-of-the-art systems are outperformed by more than 2% on the CoNLL-2014 benchmark and by 4% on JFLEG.
    </details>
+ **Language Model Based Grammatical Error Correction without Annotated Training Data**
  + Authors: Christopher Bryant, Ted Briscoe
  + Conference: NAACL
  + Link: https://aclanthology.org/W18-0529/
  + <details>
      <summary>Abstract</summary>
      Since the end of the CoNLL-2014 shared task on grammatical error correction (GEC), research into language model (LM) based approaches to GEC has largely stagnated. In this paper, we re-examine LMs in GEC and show that it is entirely possible to build a simple system that not only requires minimal annotated data (∼1000 sentences), but is also fairly competitive with several state-of-the-art systems. This approach should be of particular interest for languages where very little annotated training data exists, although we also hope to use it as a baseline to motivate future research.
    </details>
+ **Near Human-Level Performance in Grammatical Error Correction with Hybrid Machine Translation**
  + Authors: Roman Grundkiewicz, Marcin Junczys-Dowmunt
  + Conference: NAACL
  + Link: https://aclanthology.org/N18-2046/
  + <details>
      <summary>Abstract</summary>
      We combine two of the most popular approaches to automated Grammatical Error Correction (GEC): GEC based on Statistical Machine Translation (SMT) and GEC based on Neural Machine Translation (NMT). The hybrid system achieves new state-of-the-art results on the CoNLL-2014 and JFLEG benchmarks. This GEC system preserves the accuracy of SMT output and, at the same time, generates more fluent sentences as it typical for NMT. Our analysis shows that the created systems are closer to reaching human-level performance than any other GEC system reported so far.
    </details>
+ **Reference-less Measure of Faithfulness for Grammatical Error Correction**
  + Authors: Leshem Choshen, Omri Abend
  + Conference: NAACL
  + Link: https://aclanthology.org/N18-2020/
  + <details>
      <summary>Abstract</summary>
      We propose USim, a semantic measure for Grammatical Error Correction (that measures the semantic faithfulness of the output to the source, thereby complementing existing reference-less measures (RLMs) for measuring the output’s grammaticality. USim operates by comparing the semantic symbolic structure of the source and the correction, without relying on manually-curated references. Our experiments establish the validity of USim, by showing that the semantic structures can be consistently applied to ungrammatical text, that valid corrections obtain a high USim similarity score to the source, and that invalid corrections obtain a lower score.
    </details>
+ **A Reassessment of Reference-Based Grammatical Error Correction Metrics**
  + Authors: Shamil Chollampatt, Hwee Tou Ng
  + Conference: COLING
  + Link: https://aclanthology.org/C18-1231/
  + <details>
      <summary>Abstract</summary>
      Several metrics have been proposed for evaluating grammatical error correction (GEC) systems based on grammaticality, fluency, and adequacy of the output sentences. Previous studies of the correlation of these metrics with human quality judgments were inconclusive, due to the lack of appropriate significance tests, discrepancies in the methods, and choice of datasets used. In this paper, we re-evaluate reference-based GEC metrics by measuring the system-level correlations with humans on a large dataset of human judgments of GEC outputs, and by properly conducting statistical significance tests. Our results show no significant advantage of GLEU over MaxMatch (M2), contradicting previous studies that claim GLEU to be superior. For a finer-grained analysis, we additionally evaluate these metrics for their agreement with human judgments at the sentence level. Our sentence-level analysis indicates that comparing GLEU and M2, one metric may be more useful than the other depending on the scenario. We further qualitatively analyze these metrics and our findings show that apart from being less interpretable and non-deterministic, GLEU also produces counter-intuitive scores in commonly occurring test examples.
    </details>
+ **Cool English: a Grammatical Error Correction System Based on Large Learner Corpora**
  + Authors: Yu-Chun Lo, Jhih-Jie Chen, Chingyu Yang, Jason Chang
  + Conference: COLING
  + Link: https://aclanthology.org/C18-2018/
  + <details>
      <summary>Abstract</summary>
      This paper presents a grammatical error correction (GEC) system that provides corrective feedback for essays. We apply the sequence-to-sequence model, which is frequently used in machine translation and text summarization, to this GEC task. The model is trained by EF-Cambridge Open Language Database (EFCAMDAT), a large learner corpus annotated with grammatical errors and corrections. Evaluation shows that our system achieves competitive performance on a number of publicly available testsets.
    </details>
+ **A Multilayer Convolutional Encoder-Decoder Neural Network for Grammatical Error Correction**
  + Authors: Shamil Chollampatt, Hwee Tou Ng
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/12069
