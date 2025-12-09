# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)


encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN('dot', embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)

print('ready')
