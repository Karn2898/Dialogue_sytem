def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
def readVocs(datafile, corpus_name):
    print(f"Reading lines from {datafile}...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = []
    for l in lines:
        parts = l.split('\t')
        if len(parts) == 2:
            pairs.append([normalizeString(s) for s in parts])
        elif len(parts) == 1:
            pairs.append([normalizeString(parts[0]), ''])
    voc = Vocabulary(corpus_name)
    return voc, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, datafile):
    voc, pairs = readVocs(datafile, corpus_name)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:
        voc.addsentence(pair[0])
        voc.addsentence(pair[1])
    print(f"Counted words: {voc.num_words}")
    return voc, pairs