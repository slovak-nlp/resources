
## Models and Tools

### General Models

#### [Slovak Spacy Model](https://github.com/hladek/spacy-skmodel)

- Contains Floret Word Vectors.
- Tagger module uses Slovak National Corpus Tagset.
- Morphological analyzer uses Universal dependencies tagset and is trained on Slovak dependency treebank.
- Lemmatizer is trained on Slovak dependency treebank.
- Named entity recognizer is trained separately on WikiAnn database.

### Word embeddings

#### [JÚĽŠ word embeddings](https://www.juls.savba.sk/data/semä/)

* word form, POS+lemma, fasstext embeddings
* source: JÚĽŠ + SNK (prim), also from older prim-* corpora
* description: https://www.juls.savba.sk/semä.html

#### [ELMo word embeddings](https://github.com/HIT-SCIR/ELMoForManyLangs)

* source: Wikipedia, Common Crawl

#### [fastText word embeddings - Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html)

* source: Common Crawl

#### [fastText word embeddings - Wikipedia](https://fasttext.cc/docs/en/pretrained-vectors.html)

* source: Wikipedia

### Document Embeddings


#### [Language Agnostic BERT model](https://huggingface.co/setu4993/LaBSE)

- Language-agnostic BERT Sentence Encoder (LaBSE) is a BERT-based model trained for sentence embedding for 109 languages.

#### [LASER](https://github.com/facebookresearch/LASER)

- Language agnostic sentence embeddings.

#### [E5](https://huggingface.co/intfloat/e5-large-v2)

 * Multilingual document embeddings, based on Sentence Transformers.

### Transformers

#### [Slovak Mistral 7B](https://huggingface.co/slovak-nlp/mistral-sk-7b)

- is a Slovak language version of the Mistral-7B-v0.1 large language model with 7 billion parameters.
- obtained by full parameter finetuning of the Mistral-7B-v0.1 large language model with the data from the Araneum Slovacum VII Maximum web corpus.

### [Slovak T5 Base](https://huggingface.co/TUKE-KEMT/slovak-t5-base)

- Monolingual Slovak T5 model with 300 million parameters
- Trained from scratch on large web corpus

#### [SlovakBert](https://github.com/gerulata/slovakbert)

- Slovak RoBERTa base language model
- trained on web corpus

#### [sk-bert](https://github.com/Ardevop-sk/sk-bert-model)

- Slovak BERT by Ardevop SK

#### [ApoTro/slovak-t5-small](https://huggingface.co/ApoTro/slovak-t5-small)

Slovak T5 small, created by fine-tuning mT5 small.

#### [VoxPopuli Slovak model](https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-sk)

- VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation
- Facebook's Wav2Vec2 base model pretrained on the 10K unlabeled subset of VoxPopuli corpus and fine-tuned on the transcribed data in sk

#### [m-BERT](https://github.com/google-research/bert/blob/master/multilingual.md)

* multilingual BERT, trained on Wikipedia

#### [Slovak BPE Baby Language Model (SK_BPE_BLM)](https://huggingface.co/daviddrzik/SK_BPE_BLM)

- SK_BPE_BLM is a pretrained small language model for the Slovak language, based on the RoBERTa architecture. The model utilizes standard Byte-Pair Encoding (BPE) tokenization (pureBPE, more info here) and is case-insensitive, meaning it operates in lowercase.
- The SK_BPE_BLM model was pretrained using a subset of the OSCAR 2019 corpus
- Sequence length: 256 tokens
- Number of parameters: 58 million
- There are fine tuned models:
    * SK_BPE_BLM-ner: Fine-tuned for Named Entity Recognition (NER) tasks.
    * SK_BPE_BLM-pos: Fine-tuned for Part-of-Speech (POS) tagging.
    * SK_BPE_BLM-qa: Fine-tuned for Question Answering tasks.
    * SK_BPE_BLM-sentiment-csfd: Fine-tuned for sentiment analysis on the CSFD (movie review) dataset.
    * SK_BPE_BLM-sentiment-multidomain: Fine-tuned for sentiment analysis across multiple domains.
    * SK_BPE_BLM-sentiment-reviews: Fine-tuned for sentiment analysis on general review datasets.
    * SK_BPE_BLM-topic-news: Fine-tuned for topic classification in news articles.

### Translation models


#### [Helsinki Opus NLP](https://opus.nlpl.eu/Opus-MT/)

- Bidirectional translation models for Slovak for multiple languages
- Also available for HF Transformers
- Contains SentencePiece tokenization models
- For [MarianNMT](https://marian-nmt.github.io/)
- English, German, Finish, French, Swedich, 

#### [NLLB - No Langauge Left Behind](https://github.com/gordicaleksa/Open-NLLB)

- Multilingual translation model for Fairseq
- Provides also language detection models
- [Original Fairseq REPO](https://github.com/facebookresearch/fairseq/tree/nllb)
- HuggingFace [Transformers integration](https://huggingface.co/facebook/nllb-200-distilled-600M) - distilled 600M version
  

#### [MadLad400](https://huggingface.co/google/madlad400-3b-mt)

- Uses T5 architecture
- https://arxiv.org/abs/2309.04662
- Supports 400 languages, including Slovak
- Previously used for Google Translate


#### [M2M 100](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100)

- Multilingual translation model with Slovak support.
- Build for Fairseq
- [HuggingFace Transformers model](https://huggingface.co/facebook/m2m100_418M)
 
#### [Flores](https://github.com/pytorch/fairseq/tree/master/examples/flores101)

- Flores101: Large-Scale Multilingual Machine Translation
- Baseline pretrained models for small and large tracks of WMT 21 Large-Scale Multilingual Machine Translation competition.
- Includes Slovak language
- For fairseq

### Tools and demos

#### [NER web demo](https://www.juls.savba.sk/nerd/)

- models: CNEC 2.0 cs2sk, morphodita SNK
- source: JÚĽŠ SAV, SNK


#### [Lemmatization, Morphological analysis, disambiguation (tagging), web demo](https://morphodita.juls.savba.sk/)

* form, lemma, POS+MSD
* source: JÚĽŠ SAV

#### [Lemmatization, Morphological analysis, disambiguation (tagging), web API](https://www.juls.savba.sk/MSD_en.html)

* form, lemma, POS+MSD
* source: JÚĽŠ SAV

#### [Lemmatization, Morphological analysis, disambiguation (tagging), diacritics-less Slovak, web demo](https://www.juls.savba.sk/bezdiak/)

* form, lemma, POS+MSD
* source: JÚĽŠ SAV


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

#### [Semä](https://www.juls.savba.sk/semä)

* Web-based  Visualisation of Slovak word vectors

#### [Simplemma](https://pypi.org/project/simplemma/)

* Lemmatization for 25 languages
* In Python
* Slovak trained on UDP corpus

### Tokenizers

#### [Slovak Subword Tokenizers](https://github.com/daviddrzik/Slovak_subword_tokenizers)

- implementations of two tokenizers developed specifically for the Slovak language: pureBPE and SKMT
- Držík, D., & Forgac, F. (2024). Slovak morphological tokenizer using the Byte-Pair Encoding algorithm. PeerJ Computer Science, 10, e2465. https://doi.org/10.7717/peerj-cs.2465

#### [Slovak Lexe](https://github.com/hladek/slovak-lexer)

- Fast word-based tokenizers based on a state-machine.
- Can recognize sentences and words.
- Contains rules for abbreviations,adresses, dates, ordinal numbers.
