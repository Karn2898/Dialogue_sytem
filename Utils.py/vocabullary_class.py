PAD_TOKEN=0
SOS_TOKEN=1
EOS_TOKEN=2

class Vocabulary:
  def __init__(self,name):
    self.name=name
    self.trimmed=False
    self.word2index={}
    self.word2count={}
    self.index2word={PAD_TOKEN:"pad",SOS_TOKEN:"sos",EOS_TOKEN:"eos"}
    self.num_words=3

  def addsentence(self, sentence):
    for word in sentence.split(' '):
      self.addword(word)

  def addword(self,word):
    if word not in self.word2index:
      self.word2index[word]=self.num_words
      self.word2count[word]=1
      self.index2word[self.num_words]=word
      self.num_words+=1

    else:
     self.word2count[word]+=1

  def trim(self,min_count):
    if self.trimmed:
      return
    self.trimmed=True

    keep_words=[]
    for k,v in self.word2count.items():
      if v>=min_count:
        keep_words.append(k)

    print(f'Keep words: {len(keep_words)} / {len(self.word2index)} = {len(keep_words) / len(self.word2index):.4f}')

    # Reinitialize dictionaries
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
    self.num_words = 3

    for word in keep_words:
        self.addword(word)