package uy.com.collokia.nlp.parser.mate.corpus

import java.io.File

class CorpusConllConverter {
    companion object {
        @JvmStatic fun main(args: Array<String>) {

            val conllConverter = CorpusConllConverter()
            File("./data/ud-treebank/spanish/UD_Spanish-AnCora").listFiles().filter { file ->
                file.name.endsWith("conllu")
            }.forEach { file ->
                conllConverter.conllConverter(file.absolutePath)
            }
        }

    }

    fun conllConverter(fileName: String) {
        File(fileName + "conv").bufferedWriter(Charsets.UTF_8).use { writer ->

            File(fileName).forEachLine { line ->
                if (line.isEmpty()) {
                    writer.write("\n")
                } else if (!line.startsWith("#")) {
                    writer.write("$line\t_\t_\t_\t_\n")
                }
            }
        }
    }
}

