import torch
from torch import nn

# import torch.nn.functional as F


class MiniBERT(nn.Module):
    """
    A minimal BERT-like model for training Named Entity Recognition. We start with a Masked Language Model
    head for predicting masked tokens in input (see train.py), later swapped out for a NER head.
    """

    def __init__(self, vocab_size, max_seq_len, embed_size, hidden_size, n_heads, n_layers, device):
        super().__init__()
        # TODO provide padding_idx?
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, device=device)
        self.pos_embed = nn.Embedding(max_seq_len, embed_size, device=device)

        # TODO reimplementing from scratch would be interesting and we'd be able to monitor e.g. attention
        # entropy
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=n_heads, dim_feedforward=hidden_size, batch_first=True, device=device
        )

        # From the docs:
        # All layers in the TransformerEncoder are initialized with the same parameters. It is recommended to
        # manually initialize the layers after creating the TransformerEncoder instance.
        # ChatGPT claims this is fine for standard models. A default-initialised layer is close to identity
        # (small, zero-centered weights mean skip-paths dominate) and so no layer dominates, allowing them
        # to gradually diverge during training.
        # Note: no device passed here, only the layer itself has that (no weights in the encoder, I guess)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        # Encoder's internal normalisation layers stabilise gradient flow and keeps residual bocks sane.
        # Applying a norm to the output of the encoder prepares it for consumption by downstream task heads.
        self.norm = nn.LayerNorm(embed_size, device=device)

        # Masked Language Model head - once we've trained the underlying semantic model, we'll discard it
        # and add a Named Entity Recognition classifier head instead
        # TODO see if tying MLM to embeddings lowers loss floor
        self.mlm_head = nn.Linear(embed_size, vocab_size, device=device)

        self.device = device

        # When training NER, first freeze everything other than the NER head itself, train it, then unfreeze
        # and fine-tune jointly.

    def forward(self, input_seq, pad_mask):
        """
        pad_mask: True for pad tokens (not allowed to attend)
        """
        seq_len = input_seq.size(1)  # batch first
        embeddings = self.embed(input_seq)
        # normally we'd have rotational encodings or some such
        pos_embeddings = self.pos_embed(torch.arange(seq_len, device=self.device))

        # addition implicitly broadcasted across the batch
        # >>> t1
        # tensor([[0., 0., 0.],
        #         [0., 0., 0.],
        #         [0., 0., 0.]])
        # >>> t2
        # tensor([-0.8718, -0.9769, -0.5008])
        # >>> t2 + t1
        # tensor([[-0.8718, -0.9769, -0.5008],
        #         [-0.8718, -0.9769, -0.5008],
        #         [-0.8718, -0.9769, -0.5008]])
        effective_embeddings = embeddings + pos_embeddings

        # Transformer.forward() returns a hidden state per token of input.
        # Each component z_i[j] of the i'th hidden vector is the model’s score for “token j belongs here”.
        encodings = self.encoder(effective_embeddings, src_key_padding_mask=pad_mask)
        normalised = self.norm(encodings)
        return self.mlm_head(normalised)
