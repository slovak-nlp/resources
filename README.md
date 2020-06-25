# Resources
A curated list of resources for the processing of Slovak language.

## Pages

 * [Slovak resources](https://github.com/essential-data/nlp-sk-interesting-links) by Essential Data

## Tools

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
* implementation in Java/DL4J
* license: GNU AGPLv3

## Corpora, datasets, vocabularies

### Word embeddings

#### [ELMo word embeddings](https://github.com/HIT-SCIR/ELMoForManyLangs)

* source: Wikipedia, Common Crawl

#### [fastText word embeddings - Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html)

* source: Common Crawl

#### [fastText word embeddings - Wikipedia](https://fasttext.cc/docs/en/pretrained-vectors.html)

* source: Wikipedia

#### [m-BERT](https://github.com/google-research/bert/blob/master/multilingual.md)

* multilingual BERT, trained on Wikipedia

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

#### [MULTEXT-East free lexicons 4.0](https://www.clarin.si/repository/xmlui/handle/11356/1041)

* form, lemma, POS (Multext East)

### Parallel

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

### Sentiment

#### [Twitter sentiment for 15 European languages](https://www.clarin.si/repository/xmlui/handle/11356/1054)

* source: Twitter

### NER

#### [Cross-lingual Name Tagging and Linking for 282 Languages](https://elisa-ie.github.io/wikiann/)

* [download data](https://drive.google.com/drive/folders/1bkK6ly_awxe9IgAKL16VVvCtjcYcDSw8)
* automatic annotation
* source: Wikipedia

### Wordnet

#### [Slovak Wordnet](https://korpus.sk/WordNet.html)
