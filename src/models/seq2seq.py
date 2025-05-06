import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as pl
import random

# Define special token indices - adjust as per your vocabulary
PAD_IDX = 0
SOS_IDX = 1  # Start Of Sequence
EOS_IDX = 2  # End Of Sequence

class MusicEncoder(nn.Module):
    """Encodes the input music feature sequence."""
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # LSTM layer for processing music features
        # batch_first=True means input and output tensors are provided as (batch, seq, feature)
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, 
                           dropout=dropout if n_layers > 1 else 0, # Dropout only if multiple layers
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, music_sequence):
        # music_sequence: (batch_size, music_seq_len, music_feature_dim)
        
        # Pass sequence through LSTM
        # outputs: (batch_size, music_seq_len, hidden_dim) - output features from each time step
        # hidden: (n_layers, batch_size, hidden_dim) - final hidden state
        # cell: (n_layers, batch_size, hidden_dim) - final cell state
        outputs, (hidden, cell) = self.rnn(music_sequence)
        
        return outputs, hidden, cell

class Attention(nn.Module):
    """Bahdanau-style Attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        # Linear layers for calculating attention scores
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False) # For decoder hidden state
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False) # For encoder outputs
        self.Va = nn.Linear(hidden_dim, 1, bias=False)          # To produce a single score per encoder output

    def forward(self, decoder_hidden_query, encoder_outputs):
        # decoder_hidden_query: (batch_size, hidden_dim) - typically the top layer of the decoder's previous hidden state
        # encoder_outputs: (batch_size, music_seq_len, hidden_dim) - all encoder hidden states

        batch_size = encoder_outputs.size(0)
        music_seq_len = encoder_outputs.size(1)

        # Reshape decoder_hidden_query to be (batch_size, 1, hidden_dim) to broadcast across encoder_outputs
        decoder_hidden_query = decoder_hidden_query.unsqueeze(1)

        # Calculate energy scores
        # self.Wa(decoder_hidden_query) : (batch_size, 1, hidden_dim)
        # self.Ua(encoder_outputs) : (batch_size, music_seq_len, hidden_dim)
        # Summing them (broadcasts decoder_hidden_query part)
        energy = torch.tanh(self.Wa(decoder_hidden_query) + self.Ua(encoder_outputs))
        # energy: (batch_size, music_seq_len, hidden_dim)

        # Get attention scores (weights)
        # self.Va(energy) : (batch_size, music_seq_len, 1)
        attention_scores = self.Va(energy).squeeze(2) # (batch_size, music_seq_len)

        # Apply softmax to get probabilities
        attention_weights = F.softmax(attention_scores, dim=1) # (batch_size, music_seq_len)

        # Calculate context vector
        # attention_weights.unsqueeze(1) : (batch_size, 1, music_seq_len)
        # torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) : (batch_size, 1, hidden_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1) # (batch_size, hidden_dim)

        return context_vector, attention_weights

class DanceDecoderStep(nn.Module):
    """Performs a single decoding step for generating a dance label."""
    def __init__(self, dance_label_vocab_size, embed_dim, hidden_dim, n_layers, dropout, attention_module):
        super().__init__()
        self.dance_label_vocab_size = dance_label_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer for input dance labels (previous prediction or ground truth)
        self.embedding = nn.Embedding(dance_label_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = attention_module
        
        # Decoder RNN (LSTM)
        # Input to LSTM is concatenation of embedded previous label and context vector
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, n_layers, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        # Fully connected layer to map RNN output to dance label vocabulary
        self.fc_out = nn.Linear(hidden_dim, dance_label_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, decoder_hidden, decoder_cell, encoder_outputs):
        # input_token: (batch_size) - tensor of previous dance label indices
        # decoder_hidden: (n_layers, batch_size, hidden_dim) - previous decoder hidden state
        # decoder_cell: (n_layers, batch_size, hidden_dim) - previous decoder cell state
        # encoder_outputs: (batch_size, music_seq_len, hidden_dim) - outputs from encoder

        # Add a sequence dimension to input_token for embedding: (batch_size, 1)
        input_token = input_token.unsqueeze(1)
        
        # Embed the input token
        # embedded: (batch_size, 1, embed_dim)
        embedded = self.dropout(self.embedding(input_token))

        # Get the top layer of the previous decoder hidden state for attention query
        # decoder_hidden[-1] gives (batch_size, hidden_dim)
        attention_query = decoder_hidden[-1] # Assuming n_layers >= 1
        
        # Calculate context vector and attention weights
        # context_vector: (batch_size, hidden_dim)
        # attention_weights: (batch_size, music_seq_len)
        context_vector, attention_weights = self.attention(attention_query, encoder_outputs)
        
        # Reshape context_vector to (batch_size, 1, hidden_dim) to concatenate with embedded input
        context_vector = context_vector.unsqueeze(1)
        
        # Concatenate embedded input token and context vector
        # rnn_input: (batch_size, 1, embed_dim + hidden_dim)
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        
        # Pass through LSTM
        # output: (batch_size, 1, hidden_dim)
        # hidden: (n_layers, batch_size, hidden_dim)
        # cell: (n_layers, batch_size, hidden_dim)
        output, (hidden, cell) = self.rnn(rnn_input, (decoder_hidden, decoder_cell))
        
        # Get prediction (logits over dance label vocabulary)
        # output.squeeze(1): (batch_size, hidden_dim)
        # prediction: (batch_size, dance_label_vocab_size)
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell, attention_weights

class Seq2SeqDanceGenerator(pl.LightningModule):
    """Seq2Seq model for generating dance labels from music, using PyTorch Lightning."""
    def __init__(self, music_feature_dim, dance_label_vocab_size, embed_dim, hidden_dim, 
                 n_layers, dropout, learning_rate=1e-3, teacher_forcing_ratio=0.5,
                 max_output_len=500): # max_output_len for inference
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ arguments to self.hparams

        self.music_feature_dim = music_feature_dim
        self.dance_label_vocab_size = dance_label_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_p = dropout # Renamed to avoid conflict with nn.Dropout module
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_output_len = max_output_len # Default max length for generated sequence

        # Initialize components
        self.encoder = MusicEncoder(music_feature_dim, hidden_dim, n_layers, self.dropout_p)
        self.attention = Attention(hidden_dim)
        self.decoder_step = DanceDecoderStep(dance_label_vocab_size, embed_dim, hidden_dim, 
                                             n_layers, self.dropout_p, self.attention)
        
        # Loss function - CrossEntropyLoss ignores PAD_IDX in targets
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def forward(self, music_sequence, target_dance_sequence=None):
        # music_sequence: (batch_size, music_seq_len, music_feature_dim)
        # target_dance_sequence: (batch_size, target_seq_len) - ground truth dance labels (optional, for training)
        
        batch_size = music_sequence.size(0)
        
        # Determine the maximum length of the output sequence
        # If target_dance_sequence is provided (training/validation), use its length
        # Otherwise (inference), use self.max_output_len
        if target_dance_sequence is not None:
            target_len = target_dance_sequence.size(1)
        else:
            target_len = self.max_output_len

        # Tensor to store decoder's outputs for each time step
        # outputs: (batch_size, target_len, dance_label_vocab_size)
        outputs = torch.zeros(batch_size, target_len, self.dance_label_vocab_size).to(self.device)
        
        # Encode the music sequence
        # encoder_outputs: (batch_size, music_seq_len, hidden_dim)
        # hidden, cell: (n_layers, batch_size, hidden_dim)
        encoder_outputs, hidden, cell = self.encoder(music_sequence)
        
        # Initialize decoder's first input token (SOS_IDX)
        # decoder_input: (batch_size)
        decoder_input = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)
        
        # Decoder loop: generate one label at a time
        for t in range(target_len):
            # Pass current input, previous hidden/cell states, and encoder outputs to the decoder step
            # output_logit: (batch_size, dance_label_vocab_size) - raw scores for each label
            # hidden, cell: updated decoder states
            # attention_weights: (batch_size, music_seq_len) - can be stored/visualized if needed
            output_logit, hidden, cell, _ = self.decoder_step(decoder_input, hidden, cell, encoder_outputs)
            
            # Store the output logit for this time step
            outputs[:, t] = output_logit
            
            # Decide next input for the decoder
            # If training and teacher_forcing_ratio is met, use ground truth label
            use_teacher_forcing = (target_dance_sequence is not None) and \
                                  (random.random() < self.teacher_forcing_ratio)
            
            if use_teacher_forcing:
                decoder_input = target_dance_sequence[:, t]
            else:
                # Use the decoder's own prediction as the next input (greedy decoding)
                # top1: (batch_size) - indices of the predicted labels
                top1 = output_logit.argmax(1)
                decoder_input = top1
            
            # If EOS_IDX is predicted and not using teacher forcing, stop early (optional for inference)
            # This part is more relevant for variable length generation during inference.
            # For fixed length generation or training, we usually iterate for target_len.
            # if not use_teacher_forcing and (decoder_input == EOS_IDX).all() and t > 0:
            #    outputs = outputs[:, :t+1] # Trim unused part of outputs tensor
            #    break
                
        return outputs # (batch_size, target_len, dance_label_vocab_size)

    def _common_step(self, batch, batch_idx):
        # music_seq: (batch_size, music_seq_len, music_feature_dim)
        # dance_seq: (batch_size, target_seq_len) -> This is the target for loss calculation
        # dance_seq_input: (batch_size, target_seq_len) -> This is fed to decoder with teacher forcing
        # Typically, dance_seq_input is (SOS, label1, label2, ...)
        # and dance_seq (target for loss) is (label1, label2, ..., EOS)
        # For simplicity here, assume `forward` handles the SOS and teacher forcing logic internally,
        # and `target_dance_sequence` passed to `forward` is what's used for teacher forcing.
        # The actual targets for the loss function will be `dance_seq`.
        
        music_seq, dance_seq_targets = batch # Assuming batch yields music and target dance labels

        # Get model predictions (logits)
        # During training, target_dance_sequence is dance_seq_targets for teacher forcing
        # The length of generation will be dictated by dance_seq_targets.size(1)
        logits = self(music_sequence=music_seq, target_dance_sequence=dance_seq_targets)
        
        # Calculate loss
        # logits: (batch_size, target_seq_len, vocab_size)
        # dance_seq_targets: (batch_size, target_seq_len)
        # Reshape for CrossEntropyLoss:
        #  - logits to (batch_size * target_seq_len, vocab_size)
        #  - targets to (batch_size * target_seq_len)
        loss = self.criterion(logits.view(-1, self.dance_label_vocab_size), 
                              dance_seq_targets.view(-1))
        
        # Calculate accuracy (optional, simple example)
        preds = torch.argmax(logits, dim=2)
        # Consider only non-padded elements for accuracy
        non_pad_elements = (dance_seq_targets != PAD_IDX)
        correct_predictions = (preds == dance_seq_targets) & non_pad_elements
        accuracy = correct_predictions.sum().float() / non_pad_elements.sum().float()
        
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # For validation, typically turn off teacher forcing or use a fixed (often 0.0) ratio
        # Here, we can control it by passing None or the targets to self()
        # To evaluate true generation capability, pass None for target_dance_sequence
        # However, if your validation targets have fixed length, you might still pass them
        # to ensure the generation length matches.
        # For this example, let's use the same logic as training for simplicity,
        # but in a real scenario, you might want dedicated inference logic here.
        
        # Option 1: Evaluate with teacher forcing (as in _common_step)
        loss, accuracy = self._common_step(batch, batch_idx)
        
        # Option 2: Evaluate without teacher forcing (true generation)
        # music_seq, dance_seq_targets = batch
        # logits_no_tf = self(music_sequence=music_seq, target_dance_sequence=None) # Generate up to max_output_len
        # # Need to handle potential length mismatch between logits_no_tf and dance_seq_targets for loss/accuracy
        # # This requires more careful handling of padding and sequence lengths.
        # # For simplicity, we stick to Option 1 for now.

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Similar to validation_step, usually without teacher forcing.
        loss, accuracy = self._common_step(batch, batch_idx) # Using same logic for simplicity
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':
    # --- Configuration ---
    MUSIC_FEATURE_DIM = 256  # Example: Dimension of your music features (e.g., MFCCs, spectrogram features)
    DANCE_LABEL_VOCAB_SIZE = 50 # Example: Number of unique dance labels + special tokens (PAD, SOS, EOS)
    EMBED_DIM = 128          # Embedding dimension for dance labels
    HIDDEN_DIM = 256         # Hidden dimension for LSTMs (encoder and decoder)
    N_LAYERS = 2             # Number of layers for LSTMs
    DROPOUT = 0.3            # Dropout rate
    LEARNING_RATE = 0.001
    TEACHER_FORCING_RATIO = 0.5 # Probability of using teacher forcing during training
    MAX_OUTPUT_LEN = 100     # Max sequence length for generation during inference if no target is given

    # --- Instantiate Model ---
    model = Seq2SeqDanceGenerator(
        music_feature_dim=MUSIC_FEATURE_DIM,
        dance_label_vocab_size=DANCE_LABEL_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        max_output_len=MAX_OUTPUT_LEN
    )

    # --- Create Dummy Data for Testing ---
    BATCH_SIZE = 4
    MUSIC_SEQ_LEN = 200  # Length of music sequence
    DANCE_SEQ_LEN = 180  # Length of dance label sequence (can be different from music)

    # Dummy music input: (batch_size, music_seq_len, music_feature_dim)
    dummy_music_seq = torch.randn(BATCH_SIZE, MUSIC_SEQ_LEN, MUSIC_FEATURE_DIM)
    
    # Dummy target dance labels: (batch_size, dance_seq_len)
    # Labels should be integers from 0 to DANCE_LABEL_VOCAB_SIZE - 1
    # Ensure PAD_IDX, SOS_IDX, EOS_IDX are handled in your actual data preparation
    dummy_dance_labels = torch.randint(PAD_IDX + 1, DANCE_LABEL_VOCAB_SIZE, (BATCH_SIZE, DANCE_SEQ_LEN))
    # Example: Make some sequences shorter by padding
    dummy_dance_labels[0, DANCE_SEQ_LEN//2:] = PAD_IDX 
    dummy_dance_labels[1, DANCE_SEQ_LEN-10:] = PAD_IDX 

    # --- Test Forward Pass (Training mode with teacher forcing) ---
    print("Testing forward pass (training mode)...")
    # During training, target_dance_sequence is provided for teacher forcing and determining output length
    logits_train_mode = model(music_sequence=dummy_music_seq, target_dance_sequence=dummy_dance_labels)
    print("Logits shape (train mode):", logits_train_mode.shape) # Expected: (BATCH_SIZE, DANCE_SEQ_LEN, DANCE_LABEL_VOCAB_SIZE)
    assert logits_train_mode.shape == (BATCH_SIZE, DANCE_SEQ_LEN, DANCE_LABEL_VOCAB_SIZE)

    # --- Test Forward Pass (Inference mode) ---
    print("\nTesting forward pass (inference mode)...")
    # During inference, target_dance_sequence is None, model generates up to max_output_len
    model.eval() # Set model to evaluation mode (affects dropout, etc.)
    with torch.no_grad(): # Disable gradient calculations for inference
        logits_inference_mode = model(music_sequence=dummy_music_seq, target_dance_sequence=None)
    model.train() # Set model back to train mode
    print("Logits shape (inference mode):", logits_inference_mode.shape) # Expected: (BATCH_SIZE, MAX_OUTPUT_LEN, DANCE_LABEL_VOCAB_SIZE)
    assert logits_inference_mode.shape == (BATCH_SIZE, MAX_OUTPUT_LEN, DANCE_LABEL_VOCAB_SIZE)

    # --- Test Training Step (requires a DataLoader) ---
    print("\nTesting training_step (requires DataLoader)...")
    # For a full test of training_step, you'd wrap dummy data in a DataLoader
    # and use a PyTorch Lightning Trainer. This is a simplified direct call.
    
    # Create a dummy batch
    dummy_batch = (dummy_music_seq, dummy_dance_labels)
    
    # Calculate loss and accuracy using the training_step logic
    training_loss, training_accuracy = model._common_step(dummy_batch, 0)
    print(f"Dummy Training Loss: {training_loss.item()}")
    print(f"Dummy Training Accuracy: {training_accuracy.item()}")

    # --- Example of using PyTorch Lightning Trainer (minimal) ---
    # from torch.utils.data import TensorDataset, DataLoader
    #
    # # Create a dummy dataset and dataloader
    # # In a real scenario, your dataset would load actual music and dance data
    # train_dataset = TensorDataset(dummy_music_seq, dummy_dance_labels)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    #
    # val_dataset = TensorDataset(dummy_music_seq, dummy_dance_labels) # Using same data for simplicity
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    #
    # print("\nInitializing PyTorch Lightning Trainer...")
    # # trainer = pl.Trainer(max_epochs=1, fast_dev_run=True) # fast_dev_run runs 1 batch for train/val/test
    # # trainer.fit(model, train_dataloader, val_dataloader)
    # print("To run with Trainer, uncomment the lines above and ensure you have a valid dataset/dataloader.")
    print("\nSeq2SeqDanceGenerator model structure is ready.")
    print("Remember to replace placeholder dimensions and configure your DataLoader.")

