# Resources
A curated list of resources for the processing of Slovak language.

## Pages

 * [Slovak resources](https://github.com/essential-data/nlp-sk-interesting-links) by Essential Data
 * [Slovak speech and language processing ](https://nlp.kemt.fei.tuke.sk) at KEMT FEI TUKE with tools, demos and language resources.

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

#### [Slovak Universal Dependencies](https://github.com/UniversalDependencies/UD_Slovak-SNK)

* tokenization, segmentation, UPOS, XPOS (SNK), UD, lemma
* manual annotation
* format: conllu
* source: SNK

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

#### [Paracrawl](https://www.paracrawl.eu/)

 * Parallel web Corpus with Slovak Part 
 * 3.3 mil sentences English-Slovak

#### [WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)

 * Unsupervised processing of Wikipedia to obtain parallel corpora
 * Used LASER embeddings.
 * 85 different languages, 1620 language pairs, 134M parallel sentences, out of which 34M are aligned with English

### Sentiment

#### [Twitter sentiment for 15 European languages](https://www.clarin.si/repository/xmlui/handle/11356/1054)

* source: Twitter
* 3 categories - positive, negative, neutral

#### [SentiGrade](https://sentigrade.fiit.stuba.sk/data)

* Dataset contains totally 1 588 comments in Slovak language from various Facebook pages. The texts are annotated by 5 categories.

### NER

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
