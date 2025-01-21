# Resources
A curated list of resources for the processing of Slovak language.

 * [Models and tools](MODELS.md)

## Pages

 * [Slovak resources](https://github.com/essential-data/nlp-sk-interesting-links) by Essential Data
 * [Slovak speech and language processing ](https://nlp.kemt.fei.tuke.sk) at KEMT FEI TUKE with tools, demos and language resources.
 * [Slovak National Corpus](https://korpus.sk/)

## Corpora, datasets, vocabularies

### Web

#### [Multilingual C4](https://github.com/allenai/allennlp/discussions/5265)

- Created for training [mT5 model](https://www.semanticscholar.org/paper/mT5%3A-A-Massively-Multilingual-Pre-trained-Xue-Constant/74276a37bfa50f90dfae37f767b2b67784bd402a).
- Contains 67GB Slovak part. bv
- Available in Tensorflow format and HuggingFace JSON format. 
- Can be downloaded from https://github.com/allenai/allennlp/discussions/5056 using git LFS.

#### [Common Crawl](https://commoncrawl.org/2020/03/february-2020-crawl-archive-now-available/)

#### [SkTenTen](https://www.sketchengine.eu/sktenten-slovak-corpus/)

* automatic POS (SNK)
* source: web
* can be downloaded from [Clarin](https://lindat.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCDB-0?show=full)

#### [Oscar](https://oscar-corpus.com)

* deduplicated
* source: Common Crawl

#### [Aranea](http://ucts.uniba.sk/aranea_about/)

* automatic lemmatization, MSD, POS ([AUT](http://ucts.uniba.sk/aranea_about/aut.html), ensemble lemmatization+POS)
* source: web

#### [HC Corpora](http://corpora.epizy.com/corpora.html)

* no annotattion
* twitter part

#### [HPLT dataset(s)](https://hplt-project.org/datasets/v1.2)
* web corpora
* no annotation
* no deduplication
* lemmatized, POS+MSD+syntactically tagged, deduplicated corpus: https://www.juls.savba.sk/hpltskcorp.html

### Question  Answering


#### [SK QUAD](https://huggingface.co/datasets/TUKE-DeutscheTelekom/skquad)

- Manually annotated clone of SQUAD 2.0
- Contains "unanswerable questions"
- 92k items

#### [Slovak SQUAD](https://huggingface.co/datasets/TUKE-DeutscheTelekom/squad-sk)

- Machine translation of SQUAD 2.0 Database
- 140k annotated items

#### [qa2d-sk](https://huggingface.co/datasets/ctu-aic/qa2d-sk)

- Slovak version of the Question to Declarative Sentence (QA2D).
- Machine-translated using DeepL service.
- https://arxiv.org/abs/2312.10171
- 70k questions and answers

#### [Slovak BoolQ](https://huggingface.co/datasets/crabz/boolq_sk)

- 5 000 yes-no questions
- Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.
- Machine translated
- Can be used also for summarization

### Morpho-syntactic

#### [HamleDT](https://ufal.mff.cuni.cz/hamledt)

-  HArmonized Multi-LanguagE Dependency Treebank
-  Integrates  42 treebanks


#### [Korpus právnych predpisov v slovenčine](https://www.juls.savba.sk/legalcorp.html)

- Korpus obsahuje texty právnych predpisov (aktuálnych aj minulých) v slovenčine. Okrem automatickej lematizácie a morfologickej anotácie je korpus anotovaný aj syntakticky
- Citácia: GARABÍK, Radovan: Corpus of Slovak legislative documents. Jazykovedný časopis, 2022, Vol. 73, No 2, pp. 175-189.
- 45 miliónov tokenov
  
#### [Morphological vocabulary (old version)](https://korpus.sk/morphology_database.html)

* form, lemma, POS+MSD (SNK)
* source: SNK

#### [Morphological vocabulary (web demo)](https://korpus.sk/morphology_database.html)

* form, lemma, POS+MSD (SNK)
* source: SNK

#### [Lemmatization, Morphological analysis, disambiguation (tagging), web demo](https://morphodita.juls.savba.sk/)

* form, lemma, POS+MSD
* source: JÚĽŠ SAV

#### [Lemmatization, Morphological analysis, disambiguation (tagging), web API](https://www.juls.savba.sk/MSD_en.html)

* form, lemma, POS+MSD
* source: JÚĽŠ SAV

#### [Lemmatization, Morphological analysis, disambiguation (tagging), diacritics-less Slovak, web demo](https://www.juls.savba.sk/bezdiak/)

* form, lemma, POS+MSD
* source: JÚĽŠ SAV

#### [Slovak Dependency Treebank](https://lindat.cz/repository/xmlui/handle/11234/1-1822)

* tokenization, segmentation, UPOS, XPOS (SNK), UD, lemma
* manual annotation
* format: conllu, PDT tagset
* source: SNK

Reference:

- Gajdošová, Katarína; Šimková, Mária and et al., 2016,  Slovak Dependency Treebank, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-1822.

#### [Slovak Universal Dependencies](https://universaldependencies.org/treebanks/sk_snk/index.html)

A conversion of the Slovak Dependency Treebank into Universal Dependency tagset. 

* [GitHub page](https://github.com/UniversalDependencies/UD_Slovak-SNK)
* tokenization, segmentation, UPOS, XPOS (SNK), UD, lemma
* manual annotation
* format: conllu, UD tagset
* source: SNK

Reference:

- Zeman, Daniel. (2017). Slovak Dependency Treebank in Universal Dependencies. Journal of Linguistics/Jazykovedný casopis. 68. 10.1515/jazcas-2017-0048. 


#### [Artificial Treebank with Ellipsis](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2616)

* tokenization, segmentation, UPOS, XPOS (SNK), UD, lemma
* format: conllu
* source: Slovak UD, SNK


#### [MULTEXT-East free lexicons 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1041)

* form, lemma, POS (Multext East)

### Parallel


#### [ŠarišSet](https://huggingface.co/datasets/kiviki/SarisSet)

- Corpus of the Šariš dialect
- 4.7k examples.
- authors: Viktória Ondrejová and Marek Šuppa


#### [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php)

- 62 languages, 1,782 bitexts
- Slovak part contains 100 mil. tokens


#### [VoxPopuli](https://github.com/facebookresearch/voxpopuli)

* source: Europarl
* speech, vectors, language


#### [Czech-Slovak Parallel Corpus](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0006-AADF-0)

* automatic POS (SNK)
* source: Acquis, Europarl, EU-journal, EC-Europa, OPUS

#### [English-Slovak Parallel Corpus](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0006-AAE0-A)

* automatic POS (SNK)
* source: Acquis, Europarl, EU-journal, EC-Europa, OPUS

#### [English-Slovak parallel corpus of texts from The Ministry of Justice of the Slovak Republic](https://hdl.handle.net/21.11129/0000-000D-F908-2)

* Dataset of various English-Slovak legal texts within agenda of the Ministry,
* plain text format alligned at the sentence level, the size: 112580 words.
* It was converted into a 2895-TUs English-Slovak resource in TMX format.

#### [English-Slovak parallel corpus of texts from The Ministry of Culture of the Slovak Republic](https://data.europa.eu/data/datasets/elrc_329?locale=en)

* Dataset of various English-Slovak legal texts within agenda of the Ministry,
* plain text format alligned at the sentence level,
* the size: 105791 words
* It is converted into a 2609-TUs English-Slovak resource in TMX format.


#### [Paracrawl](https://www.paracrawl.eu/)

 * Parallel web Corpus with Slovak Part 
 * 3.3 mil sentences English-Slovak

#### [WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)

 * Unsupervised processing of Wikipedia to obtain parallel corpora
 * Used LASER embeddings.
 * 85 different languages, 1620 language pairs, 134M parallel sentences, out of which 34M are aligned with English

### Semantic textual similarity

#### [STSB-sk](https://huggingface.co/datasets/crabz/stsb-sk/tree/main)

- Machine translated by OPUS-en-sk model
- Sentence similarity dataset contains two sentences with a floating-point number between 0 and 5 as a target, where the highest number means higher similarity. The dataset contains train: 5 749, validation: 1 500 and test: 1 379 examples.
- Referenced from this [report](https://huggingface.co/ApoTro/slovak-t5-small/resolve/main/SlovakT5_report.pdf) by J. Agarský.

### Sentiment

#### [Sentiment Analysis Data for the Slovak Language](https://huggingface.co/datasets/DGurgurov/slovak_sa)

- 5k items
- positive and negative class
- Reference: Samuel Pecar, Marian Simko, and Maria Bielikova. 2019. [Improving Sentiment Classification in Slovak Language](https://aclanthology.org/W19-3716/). In Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing, pages 114–119, Florence, Italy. Association for Computational Linguistics.

#### [Slovak_sentiment](https://huggingface.co/datasets/sepidmnorozy/Slovak_sentiment)

- Unknown/undocumented source
- positive/negative

#### [Twitter sentiment for 15 European languages](https://www.clarin.si/repository/xmlui/handle/11356/1054)

* source: Twitter
* 3 categories - positive, negative, neutral

#### [SentiGrade](https://sentigrade.fiit.stuba.sk/data)

* Dataset contains totally 1 588 comments in Slovak language from various Facebook pages. The texts are annotated by 5 categories.

#### [STS2-sk](https://github.com/kinit-sk/slovakbert-auxiliary/tree/main/sentiment_reviews)

- Machine translated
- Sentiment analysis dataset, binary classification task: positive sentiment, negative sentiment.  It includes reviews from 7 categories with positive, neutral and negative sentiment labels. 
- Source: [Slovakbert auxiliary repository](https://github.com/kinit-sk/slovakbert-auxiliary/tree/main/sentiment_reviews) BY Matúš Pikuliak, Štefan Grivalský, Martin Konôpka, Miroslav Blšták, Martin Tamajka, Viktor Bachratý, Marián Šimko, Pavol Balážik, Michal Trnka, and Filip Uhlárik. , 2021
- Referenced from this [report](https://huggingface.co/ApoTro/slovak-t5-small/resolve/main/SlovakT5_report.pdf) by J. Agarský.

#### [sk csfd movie reviews](https://huggingface.co/datasets/fewshot-goes-multilingual/sk_csfd-movie-reviews)

- CSFD Movie Reviews
- 25k items

### Hate Speech

#### [Hate Speech Slovak](https://huggingface.co/datasets/TUKE-KEMT/hate_speech_slovak)

- 13k items
- Crowdsourced hate and offensive speech in Facebook comments
- binary classification


### Fact Checking

#### [MultiClaim](https://github.com/kinit-sk/multiclaim)

- Multilingual fact checking database with Slovak part
- Contains 28k posts in 27 languages from social media, 206k fact-checks in 39 languages written by professional fact-checkers, as well as 31k connections between these two groups.


#### [Demagog](https://corpora.kiv.zcu.cz/fact-checking/) 

9.1k Czech, 2.8k Polish and 12.6k Slovak labeled claims with reasoning: demagog.zip (~16.5 MB)

#### [qacg-sk](https://huggingface.co/datasets/ctu-aic/qacg-sk)

- Machine translated facts with evidence representend as references to Wikipedia pages.
- 350k items


### Biases

#### [GEST](https://github.com/kinit-sk/gest)

- GEST dataset used to measure gender-stereotypical reasoning in language models and machine translation systems.


### Instructions

#### [SlovAlpaca](https://huggingface.co/datasets/blip-solutions/SlovAlpaca)

- Machine translation of the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- 40k annotations

### Named Entity Recognition

#### [CNEC 2.0 cs2sk](https://www.juls.savba.sk/ner_en.html)

- CNEC 2.0 Czech model machine translated to Slovak & filtered
- CNEC entity hierarchy
- source: JÚĽŠ SAV

#### [NER web demo](https://www.juls.savba.sk/nerd/)

- models: CNEC 2.0 cs2sk, morphodita SNK
- source: JÚĽŠ SAV, SNK

#### [Universal NER Slovak](https://huggingface.co/datasets/universalner/uner_llm_inst_slovak)

- 8,48k sentences
- Annotated by a large langauge model
- PER, ORG, LOC annotations

#### [Universal NER (UNER) Slovak SNK](https://github.com/UniversalNER/UNER_Slovak-SNK)

- 8,48k train, 1k dev and 2k test sentences from Universal Dependencies
- Human annotated by 2 annotators as part of the [Universal NER](https://www.universalner.org/) project
- PER, ORG, LOC annotations

#### [WikiGold](https://huggingface.co/datasets/NaiveNeuron/wikigoldsk)

- 10k manually annotated items from Wikipedia

#### [ju-bezdek/conll2003-SK-NER](https://huggingface.co/datasets/ju-bezdek/conll2003-SK-NER)

 translated version of the original CONLL2003 dataset (translated from English to Slovak via Google translate


#### [Polyglot NER](https://huggingface.co/datasets/polyglot_ner)

- automatically annotated Wikipedia for Named Entities
- massively multilingual
- Slovak part has 500k sentences.
- Reference: Al-Rfou, Rami, et al. "Polyglot-NER: Massive multilingual named entity recognition." Proceedings of the 2015 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2015.

#### [Cross-lingual Name Tagging and Linking for 282 Languages](https://elisa-ie.github.io/wikiann/)

* [download data](https://drive.google.com/drive/folders/1bkK6ly_awxe9IgAKL16VVvCtjcYcDSw8)
* automatic annotation
* source: Wikipedia


#### [Contextualized Language Model-based Named Entity Recognition in Slovak Texts](https://github.com/vidosuba/slovak-NER)

* Manually annotated set
* Diploma thesis at Commeius University
* PER, ORG, LOC, MISC annotations
* cca 7k sentences.


#### [DaMuEL](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5047)

* A Large Multilingual Dataset for Entity Linking
* Slovak part has 41.0M tokens and 1366.4k entities
* named entity types (PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART,MANUFACTURED


### Spelling

#### [CHIBI](https://www.juls.savba.sk/errcorp.html)

* corpus of spelling errors created from edits in Wikipedia
* spelling errors are sorted into 5 categories,

### Wordnet

#### [Slovak Wordnet](https://korpus.sk/korpusy-a-databazy/databazy/wordnet/)

### Summarization

#### [Eur Lex Sum](https://huggingface.co/datasets/dennlinger/eur-lex-sum/viewer/slovak)

- Multilingual Summarization Dataset
- Slovak part has 1.3k rows.

#### [A Summarization Dataset of Slovak News Articles](https://huggingface.co/datasets/kiviki/SlovakSum)

- 200k of news article summaries
- Reference: Marek Suppa and Jergus Adamec. 2020. [A Summarization Dataset of Slovak News Articles](https://aclanthology.org/2020.lrec-1.830/). In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 6725–6730, Marseille, France. European Language Resources Association.

#### [Slovak Gigaword](https://huggingface.co/datasets/Plasmoxy/gigatrue-slovak)


- Synthetic [Gigaword](https://huggingface.co/datasets/Harvard/gigaword) dataset translated to Slovak using [SeamlessM4T-v2](https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2).
- 3.78 mil. summaries
- Same as https://huggingface.co/datasets/Plasmoxy/gigatrue but translated to Slovak.


### Reasoning

#### [multimodal-SK-exams](https://huggingface.co/datasets/dokato/multimodal-SK-exams)

- Part of [INCLUDE](https://arxiv.org/abs/2411.19799)
- 131K manually created math questions. One choice of 4 is correct.

