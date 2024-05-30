import torch
import torch.nn as nn

from layers import PositionalEncoding, EncoderLayer


class SimplifiedBERT(nn.Module):
    """
    Simplified BERT model for sequence labeling tasks.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward network.
        num_layers (int): Number of encoder layers.
        max_len (int): Maximum length of input sequences.
        num_labels (int): Number of output labels.
        dropout (float): Dropout rate. Default is 0.1.
    """
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len,
        num_labels,
        dropout=0.1,
    ):
        super(SimplifiedBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, segment_ids, attention_mask):
        """
        Forward pass for the SimplifiedBERT model.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs.
            segment_ids (torch.Tensor): Tensor containing segment IDs.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Logits for each token in the input sequence.
        """
        token_embeddings = self.embedding(input_ids)
        segment_embeddings = self.segment_embedding(segment_ids)
        position_embeddings = self.positional_encoding(token_embeddings)

        # Combine embeddings
        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, attention_mask)

        logits = self.fc(embeddings)
        return logits
