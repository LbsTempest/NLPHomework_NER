import torch
import torch.nn as nn

from layers import PositionalEncoding, DecoderLayer


class Decoder(nn.Module):
    """
    Decoder model for sequence generation tasks.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        num_layers (int): Number of decoder layers.
        max_len (int): Maximum length of input sequences.
        output_dim (int): Dimension of the output.
        dropout (float): Dropout rate. Default is 0.1.
    """
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, output_dim, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the Decoder model.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Logits for each token in the input sequence.
        """
        # Compute token and position embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.positional_encoding(token_embeddings)

        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            embeddings = layer(embeddings, embeddings, attention_mask, attention_mask)

        logits = self.fc(embeddings)
        return logits
