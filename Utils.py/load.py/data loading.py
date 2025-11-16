corpus_name = "/content/cornell_movie_dialogs_corpus.zip"
datafile = "/content/movie_lines.txt"

voc, pairs = loadPrepareData(corpus_name, datafile)

# Trim vocabulary
voc.trim(MIN_COUNT)

!unzip /content/cornell_movie_dialogs_corpus.zip -d /content/