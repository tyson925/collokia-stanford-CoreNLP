# collokia-CoreNLP

Apache Spark-based Natural Language Processing approach to automatically process raw texts on different languages. It applies openNLP sentence detecter and tokenizer. Then, you can `lemmatize`, `n-gram`, `POS tagging` and `dependency parsing` the tokenized contents and you may store the results into ElasticSearch if you want. The Lemmatizer, POSTagger and Dependency Parser based on Mate models.

Example of Spanish tokenization:

`val tokenizer = OpenNlpTokenizer(sparkSession, inputColName = TestDocument::content.name,language =  LANGUAGE.SPANISH, isOutputRaw = false)

val tokenized = tokenizer.transform(testCorpus)

tokenized.show(10,false)`

Portuguese lemmatization:

`val lemmatizer = MateLemmatizerRaw(sparkSession, language = LANGUAGE.PORTUGUESE, isRawOutput = true)

val lemmatizedDataset = lemmatizer.transform(testCorpus)`

English POS tagging

`val tagger = MateTagger(sparkSession, language = LANGUAGE.ENGLISH, inputColName = TestDocument::content.name)

val taggedContent = tagger.transform(testCorpus)`

English Dependency parsing

`val parser = MateParser(sparkSession, language = LANGUAGE.ENGLISH,inputColName = TestDocument::content.name)
val parsedContent = parser.transform(testCorpus)`

