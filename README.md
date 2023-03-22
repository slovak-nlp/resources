# Resources
A curated list of resources for the processing of Slovak language.

## Pages

 * [Slovak resources](https://github.com/essential-data/nlp-sk-interesting-links) by Essential Data
 * [Slovak speech and language processing ](https://nlp.kemt.fei.tuke.sk) at KEMT FEI TUKE with tools, demos and language resources.
 * [Slovak National Corpus](https://korpus.sk/)

## Tools


#### [Slovak Hunspell](https://github.com/sk-spell/hunspell-sk)

- Spelling Dictionary
- List of common names, abbreviations, pejoratives and neologisms.

#### [Stanza](https://github.com/stanfordnlp/stanza)

* tokenization, segmentation, UPOS, XPOS (SNK), lemmatization, UD
* models trained on UD
* implementation in Python/PyTorch, command-line interface, web service interface
* license: Apache v2.0

#### [NLP-Cube](https://github.com/adobe/NLP-Cube)

* tokenization, segmentation, UPOS, XPOS (SNK), lemmatization, UD
* models trained on UD
* implementation in Python/dyNET, command-line interface, web service interface
* license: Apache v2.0

#### [Slovak Elasticsearch](https://github.com/essential-data/elasticsearch-sk)

* tokenization, stemming

#### [Slovak lexer](https://github.com/hladek/slovak-lexer)

* tokenization, segmentation
* implementation in C++
* license: GPL v3.0

#### [dl4dp](https://github.com/peterbednar/dl4dp)

* UPOS, UD
* models trained on UD
* implementation in Python/PyTorch, command-line interface
* license: MIT

#### [UDPipe](https://ufal.mff.cuni.cz/udpipe)

* tokenization, segmentation, UPOS, XPOS (SNK), lemmatization, UD
* models trained on UD
* implementation in C++, bindings in Java, Python, Perl, C#, command-line interface, web service interface
* license: MPL v2.0

#### [NLP4SK](http://arl6.library.sk/nlp4sk/)

* tokenization, stemming, lemmatization, diacritic restoration, POS (SNK), NER
* web service interface only
* license: ?

#### [NLP Tools](https://github.com/Ardevop-sk/nlp-tools)

* tokenization, segmentation, lemmatization, POS (OpenNLP, SNK), UD (CoreNLP), NER
* web interface at http://nlp.bednarik.top/
* Swagger REST API
* implementation in Java/DL4J
* source codes available
* license: GNU AGPLv3

### [Semä](https://www.juls.savba.sk/semä)

* Web-based  Visualisation of Slovak word vectors

### [Simplemma](https://pypi.org/project/simplemma/)

* Lemmatization for 25 languages
* In Python
* Slovak trained on UDP corpus

## Corpora, datasets, vocabularies


### Web

#### [Common Crawl](https://commoncrawl.org/2020/03/february-2020-crawl-archive-now-available/)

#### [SkTenTen](https://www.sketchengine.eu/sktenten-slovak-corpus/)

* automatic POS (SNK)
* source: web

#### [Oscar](https://oscar-corpus.com)

* deduplicated
* source: Common Crawl

#### [Aranea](http://ucts.uniba.sk/aranea_about/)

* automatic POS ([AUT](http://ucts.uniba.sk/aranea_about/aut.html), TreeTagger)
* source: web

#### [HC Corpora](http://corpora.epizy.com/corpora.html)

* no annotattion
* twitter part

### Morpho-syntactic

#### [Slovak Dependency Treebank](https://lindat.cz/repository/xmlui/handle/11234/1-1822)

* tokenization, segmentation, UPOS, XPOS (SNK), UD, lemma
* manual annotation
* format: conllu, PDT tagset
* source: SNK

Reference:

- Gajdošová, Katarína; Šimková, Mária and et al., 2016,  Slovak Dependency Treebank, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-1822.

#### [Slovak Universal Dependencies](https://universaldependencies.org/treebanks/sk_snk/index.html
)

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

#### [Morphological vocabulary](https://korpus.sk/morphology_database.html)

* form, lemma, POS (SNK)
* source: SNK

#### [Morphological vocabulary](http://tvaroslovnik.ics.upjs.sk)

* form, lemma, POS

#### [MULTEXT-East free lexicons 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1041)

* form, lemma, POS (Multext East)

### Parallel

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

#### [MULTEXT-East "1984" annotated corpus](https://www.clarin.si/repository/xmlui/handle/11356/1043)

* sentence aligned, POS
* Bulgarian, Czech, English, Estonian, Hungarian, Macedonian, Persian, Polish, Romanian, Serbian, Slovak, Slovenian
* source: "1984" novel

Reference:

Erjavec, Tomaž; et al., 2010,   MULTEXT-East "1984" annotated corpus 4.0, Slovenian language resource repository CLARIN.SI, ISSN 2820-4042,   http://hdl.handle.net/11356/1043.


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
- [Source](Slovakbert auxiliary repository. https://github.com/kinit-sk/slovakbert-auxiliary/tree/main/sentiment_reviews) BY Matúš Pikuliak, Štefan Grivalský, Martin Konôpka, Miroslav Blšták, Martin Tamajka, Viktor Bachratý, Marián Šimko, Pavol Balážik, Michal Trnka, and Filip Uhlárik. , 2021
- Referenced from this [report](https://huggingface.co/ApoTro/slovak-t5-small/resolve/main/SlovakT5_report.pdf) by J. Agarský.

### Fact Checking

#### [Demagog](https://corpora.kiv.zcu.cz/fact-checking/) 

9.1k Czech, 2.8k Polish and 12.6k Slovak labeled claims with reasoning: demagog.zip (~16.5 MB)

### NER

### [ju-bezdek/conll2003-SK-NER](https://huggingface.co/datasets/ju-bezdek/conll2003-SK-NER)

 translated version of the original CONLL2003 dataset (translated from English to Slovak via Google translate


#### [Cross-lingual Name Tagging and Linking for 282 Languages](https://elisa-ie.github.io/wikiann/)

* [download data](https://drive.google.com/drive/folders/1bkK6ly_awxe9IgAKL16VVvCtjcYcDSw8)
* automatic annotation
* source: Wikipedia

### Spelling

#### [CHIBI](https://www.juls.savba.sk/errcorp.html)

* corpus of spelling errors created from edits in Wikipedia
* spelling errors are sorted into 5 categories,

### Wordnet

#### [Slovak Wordnet](https://korpus.sk/WordNet.html)

## Models

### [Slovak Spacy Model](https://github.com/hladek/spacy-skmodel)

- Contains Floret Word Vectors.
- Tagger module uses Slovak National Corpus Tagset.
- Morphological analyzer uses Universal dependencies tagset and is trained on Slovak dependency treebank.
- Lemmatizer is trained on Slovak dependency treebank.
- Named entity recognizer is trained separately on WikiAnn database.

### Word embeddings

#### [ELMo word embeddings](https://github.com/HIT-SCIR/ELMoForManyLangs)

* source: Wikipedia, Common Crawl

#### [fastText word embeddings - Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html)

* source: Common Crawl

#### [fastText word embeddings - Wikipedia](https://fasttext.cc/docs/en/pretrained-vectors.html)

* source: Wikipedia

### Transformers

#### [SlovakBert](https://github.com/gerulata/slovakbert)

- Slovak RoBERTa base language model
- trained on web corpus

#### [sk-bert](https://github.com/Ardevop-sk/sk-bert-model)

- Slovak BERT by Ardevop SK

#### [ApoTro/slovak-t5-small](https://huggingface.co/datasets/ju-bezdek/conll2003-SK-NER)

T5 small, trained on OSCAR dataset.


#### [HuggingFace Translation models](https://huggingface.co/models?filter=sk)

- Transformer models for machine translation
- Slovak, English, Finish, Swedish, Spanish, French

#### [VoxPopuli Slovak model](https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-sk)

- VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation
- Facebook's Wav2Vec2 base model pretrained on the 10K unlabeled subset of VoxPopuli corpus and fine-tuned on the transcribed data in sk

#### [m-BERT](https://github.com/google-research/bert/blob/master/multilingual.md)

* multilingual BERT, trained on Wikipedia

#### [Language Agnostic BERT model](https://huggingface.co/setu4993/LaBSE)

- Language-agnostic BERT Sentence Encoder (LaBSE) is a BERT-based model trained for sentence embedding for 109 languages.

### [Flores](https://github.com/pytorch/fairseq/tree/master/examples/flores101)

- Flores101: Large-Scale Multilingual Machine Translation
- Baseline pretrained models for small and large tracks of WMT 21 Large-Scale Multilingual Machine Translation competition.
- Includes Slovak language
- For fairseq

### [LASER](https://github.com/facebookresearch/LASER)

- Language agnostic sentence embeddings.
