# Initialize optimizers
print('Building optimizers...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# Training iterations
print("Starting Training!")
for iteration in range(1, n_iteration + 1):
    training_batch = batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                 decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)

    if iteration % print_every == 0:
        print(f'Iteration: {iteration}; Percent complete: {iteration / n_iteration * 100:.1f}%; Average loss: {loss:.4f}')

    if iteration % save_every == 0:
        directory = 'save/model'
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'iteration': iteration,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
            'embedding': embedding.state_dict()
        }, os.path.join(directory, f'{iteration}_checkpoint.tar'))
