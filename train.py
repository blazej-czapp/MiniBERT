"""Trainer for a BERT model - first with masking, later for named entity recognition"""

import os
import json
import random

from transformers import BertTokenizerFast

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, IterableDataset

from minibert import MiniBERT

PRETRAIN_DATA_DIR = "./pretrain_data"
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
TRAIN_MASK_PROB = 0.15


class JsonDataset(IterableDataset):
    """
    Adapter for JSON training data downloaded with datatrove (see get_training_data.py)
    """

    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            with open(json_file, encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    yield sample["text"]


if __name__ == "__main__":
    cuda0 = torch.device("cuda:0")

    dataset = JsonDataset(
        [os.path.join(PRETRAIN_DATA_DIR, datafile) for datafile in os.listdir(PRETRAIN_DATA_DIR)]
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # cached at ~/.cache/huggingface
    # English by default
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased", unk_token="<unk>")

    # >>> tokenizer("hello")
    # {'input_ids': [101, 7592, 102], 'token_type_ids': [0, 0, 0], 'attention_mask': [1, 1, 1]}

    # Token_type_ids is there to distinguish tokes from separate inputs when calling tokenizer("hello", "world"):
    # >>> tokenizer("hello", "world")
    # {'input_ids': [101, 7592, 102, 2088, 102], 'token_type_ids': [0, 0, 0, 1, 1], 'attention_mask': [1, 1, 1, 1, 1]}
    # this could be e.g. a question and answer distinction
    # attention_mask distinguishes pad tokens from genuine ones, useful when padding is enabled to allow

    # 101 is the CLS (classifier) token:
    # >>> tokenizer.decode(101)
    # '[CLS]'
    # The [CLS] token is a special token used in natural language processing models, particularly those based
    # on the Transformer architecture, to represent the entire input sequence for classification tasks.
    # It is placed at the beginning of the input and helps the model capture context for making predictions.

    # BERT assigns different segment embeddings to tokens before and after [SEP]. This helps the model
    # distinguish between different text parts:
    # >>> tokenizer.decode(tokenizer("sentence one", "sentence two")['input_ids'])
    # '[CLS] sentence one [SEP] sentence two [SEP]'
    # >>> tokenizer("sentence one", "sentence two")
    # {'input_ids': [101, 6251, 2028, 102, 6251, 2048, 102], 'token_type_ids': [0, 0, 0, 0, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

    # special tokens can be accessed via tokenizer._special_tokens_map

    # We're batching through emitting tokenizations of equal sizes:
    # >>> tokenizer(["sentence one", "sentence two is longer than sentence one"], padding=True)
    # {'input_ids': [[101, 6251, 2028, 102, 0, 0, 0, 0, 0], [101, 6251, 2048, 2003, 2936, 2084, 6251, 2028, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    # (note we're passing a list now so type ids are the same, but now attention mask is 0 for pad tokens)

    model = MiniBERT(
        vocab_size=len(tokenizer.get_vocab()),
        max_seq_len=MAX_SEQ_LEN,
        embed_size=512,
        hidden_size=1024,
        n_heads=4,
        n_layers=4,
        device=cuda0,
    )

    # The optimizer isn't aware of the loss function at all - the loss updates tensor gradients by itself,
    # the optimizer then steps in and updates parameters appropriately (taking into account things like
    # learning rate, momentum, decay etc.)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # load training data

    MASK_ID = 103
    if MASK_ID not in tokenizer("[MASK]")["input_ids"]:
        raise Exception("Unknown [MASK] token ID")

    PAD_ID = 0
    if (last := tokenizer("foo", padding="max_length", max_length=6)["input_ids"][-1]) != PAD_ID:
        raise Exception(f"Unknown [PAD] token ID, expected {PAD_ID}, got {last}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    tb_writer = SummaryWriter("runs/embed_512_heads_4_layers_4_lr_00001")

    step: int = 0
    for epoch in range(10):
        print(f"Epoch: {epoch}")
        for text_batch in dataloader:
            optimizer.zero_grad()
            # TODO slide a window over long input sequences?
            batch = tokenizer(
                text_batch,
                padding="max_length",  # pad everything to the length indicated by max_length argument
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",  # return pytorch tensors rather than python lists (another option is 'np' for numpy)
            )["input_ids"]

            batch = batch.to(cuda0)

            # detach() doesn't actually copy the data but is called first to prevent cloning of gradient data
            batch_unmasked = batch.detach().clone()

            # Randomly select ~15% of tokens
            # Of those:
            # 80% -> [MASK]
            # 10% -> random token
            # 10% -> unchanged

            # Create a mask for 15% of tokens
            batch_size, seq_len = batch.shape
            mask = torch.rand(batch_size, seq_len) < TRAIN_MASK_PROB

            # Don't mask special tokens (101=[CLS], 102=[SEP], 0=[PAD])
            # 103=[MASK] shouldn't be appearing in tokenized input
            # 30522=[UNK] should probably undergo the same treatmet as others, but maybe not?
            special_tokens = {0, 101, 102}
            for i in range(batch_size):
                for j in range(seq_len):
                    if batch[i, j].item() in special_tokens:
                        mask[i, j] = False

            for i in range(batch_size):
                for j in range(seq_len):
                    if mask[i, j]:
                        rand_val = random.random()
                        if rand_val < 0.8:  # 80% -> [MASK]
                            batch[i, j] = MASK_ID
                        elif rand_val < 0.9:  # 10% -> random non-special token
                            while True:
                                rand_tok = random.randint(0, len(tokenizer.get_vocab()) - 1)
                                if rand_tok not in special_tokens:
                                    break
                            batch[i, j] = rand_tok
                        # else: 10% -> unchanged

            input_seq = batch

            pad_mask = batch == PAD_ID
            y = model(input_seq=batch, pad_mask=pad_mask)

            # Run the loss function for, and only for, every masked token. This trains the model to predict
            # masked tokens, and, eventually, all tokens (since a masked token may actually be left unchanged).
            # Passing token IDs, not one-hot vectors as targets - CrossEntropy is aware of this.
            # Gradients are averaged per batch.
            loss = loss_fn(y[mask], batch_unmasked[mask])
            loss.backward()
            optimizer.step()

            # ----- Tensorboard -----

            tb_writer.add_scalar("Loss", loss.item(), step)

            # could also compute loss per token and plot masked vs unmasked loss to make sure the model is
            # not cheating with identity (esp. if we change the masking strategy)

            # detect loss spikes with learning rate adjustments, esp. after warmup, and other instabilities
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

            # save time and memory by not computing graph updates (even if we discard them in the end by not
            # calling backward(), graph state is still computed)
            with torch.no_grad():
                # compute accuracy of actual predictions - if loss goes down while accuracy stays flat, it
                # suggests a capacity or representation bottleneck
                preds = y.argmax(dim=-1)
                correct = (preds[mask] == batch_unmasked[mask]).float().mean()
                tb_writer.add_scalar("mlm/accuracy_masked", correct.item(), step)

                # Plot global *gradient* norm:
                #   - exploding gradient presage loss divergence
                #   - vanishing norms indicate learning death
                #   - no long-term upward trend is healthy
                # using large max_norm so that we're just measuring, not really clipping
                tb_writer.add_scalar(
                    "norm/gradient", torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9), step
                )

                # feels like an overkill - gradient norms should corelate here?
                # param_norm = torch.sqrt(sum(p.norm()**2 for p in model.parameters()))
                # tb_writer.add_scalar("params/global_norm", param_norm.item(), global_step)

                # log representation scale, observe if it doesn't suddenly collapse, grow indefinitely
                # (underregularisation) or wildly oscilate
                # TODO could do the same for attention (difficult to access) and embeddings and, for deeper
                # stacks, early, mid and late
                tb_writer.add_scalar("repr/mlp_out_norm_l3", y[mask].norm().item(), step)

            step += 1

    tb_writer.close()
