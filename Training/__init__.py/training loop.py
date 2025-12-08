import torch
import random

PAD_token = 0
SOS_token = 1
EOS_token = 2


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths need to be on CPU for pack_padded_sequence usually
    lengths = lengths.to("cpu")

    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    # Clip gradients
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    # RETURN THE AVERAGE LOSS
    return sum(print_losses) / n_totals


print('Building optimizers...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

encoder.train()
decoder.train()

best_loss = float('inf')

print("Starting Training!")

for iteration in range(1, n_iteration + 1):
    training_batch = batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                 decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)

    if iteration % print_every == 0:
        print(
            f'Iteration: {iteration}; Percent complete: {iteration / n_iteration * 100:.1f}%; Average loss: {loss:.4f}')

    # Save best model
    if loss < best_loss:
        best_loss = loss
        save_checkpoint(iteration, encoder, decoder, encoder_optimizer, decoder_optimizer, loss, voc, embedding,
                        filename='best_checkpoint.tar')


