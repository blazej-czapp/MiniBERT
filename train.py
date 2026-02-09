"""Trainer for a BERT model - first with masking, later for named entity recognition"""

import os
import json
import random
import signal
import threading
import argparse
import glob
import re
import sys
from itertools import islice

from transformers import BertTokenizerFast

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, IterableDataset

from minibert import MiniBERT

PRETRAIN_DATA_DIR = "./pretrain_data"
MAX_SEQ_LEN = 128
MAX_STEPS = 120000
BATCH_SIZE = 32
NUM_EPOCHS = 20
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


def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))


stop_requested = threading.Event()


def handle_sigint(signum, frame):
    stop_requested.set()


def save_checkpoint(model, optimizer, scheduler, epoch, epoch_size, step, run_name):
    filename = os.path.join("runs", run_name, f"checkpoint_{step}.json")
    torch.save(
        {
            "epoch": epoch,
            "epoch_size": epoch_size,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        filename,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, weights_only=True)

    model = init_model(device)
    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)

    model.load_state_dict(checkpoint["model_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # has to be called after the scheduler is initialised ("calling it beforehand will overwrite the loaded
    # learning rates")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    epoch_size = checkpoint["epoch_size"]
    step = checkpoint["step"]

    return model, optimizer, scheduler, epoch, epoch_size, step


def find_latest_checkpoint(run_name):
    pattern = os.path.join("runs", run_name, "checkpoint_*.json")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    max_step = -1
    latest = None
    for ck in checkpoints:
        m = re.search(r"checkpoint_(\d+)\.json$", ck)
        if m:
            s = int(m.group(1))
            if s > max_step:
                max_step = s
                latest = ck
    return latest


def checkpoint_path_for_step(run_name, step):
    path = os.path.join("runs", run_name, f"checkpoint_{step}.json")
    return path if os.path.exists(path) else None


def init_model(device):
    return MiniBERT(
        vocab_size=len(tokenizer.get_vocab()),
        max_seq_len=MAX_SEQ_LEN,
        embed_size=512,
        hidden_size=1024,
        n_heads=4,
        n_layers=4,
        device=device,
    )


def init_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.0001)


def init_scheduler(optimizer):
    switchover = int(MAX_STEPS * 0.5)
    warmup = LinearLR(optimizer, start_factor=1.0, total_iters=switchover)
    cosine = CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=1e-6)
    # TODO try torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[switchover])
    return warmup

def seq_len_curriculum(step):
    if step < 6000:
        return 64
    elif step < 12000:
        return int(MAX_SEQ_LEN * 0.5)
    else:
        return MAX_SEQ_LEN


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    device = torch.device("cuda:0")

    dataset = JsonDataset(
        [os.path.join(PRETRAIN_DATA_DIR, datafile) for datafile in os.listdir(PRETRAIN_DATA_DIR)]
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # cached at ~/.cache/huggingface
    # English by default
    # TODO replace with a cased tokenizer for NRE
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

    parser = argparse.ArgumentParser(description="Train MiniBERT")
    parser.add_argument(
        "--run-name",
        "-r",
        help="Name for this run (checkpoint directory under runs/)",
        required=True,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume-latest", action="store_true", help="Resume from the latest checkpoint in runs/<run_name>"
    )
    group.add_argument(
        "--resume-step",
        type=int,
        help="Resume from a specific checkpoint step number (checkpoint_<step>.json)",
    )
    args = parser.parse_args()
    run_name = args.run_name

    # Decide whether to resume from checkpoint
    if os.path.exists(f"runs/{run_name}") and (args.resume_latest or args.resume_step is not None):
        if args.resume_step is not None:
            ck_path = checkpoint_path_for_step(run_name, args.resume_step)
            if ck_path is None:
                print(f"No checkpoint for step {args.resume_step} in runs/{run_name}", file=sys.stderr)
                sys.exit(1)
        else:
            ck_path = find_latest_checkpoint(run_name)
            if ck_path is None:
                print(f"No checkpoints found in runs/{run_name}", file=sys.stderr)
                sys.exit(1)

        restoring_from_checkpoint = True
        model, optimizer, scheduler, start_epoch, epoch_size, step = load_checkpoint(ck_path, device)
        print(f"Run restored from checkpoint at step {step}, epoch {start_epoch}")
        data_it = iter(dataloader)

        # we only find out the epoch size oncethe first epoch is complete
        assert start_epoch == 0 or epoch_size is not None
        skip = step % epoch_size if epoch_size is not None else step
        # skip the dataloader iterator to the resumed step in the epoch
        data_it = islice(data_it, skip, None)
    else:
        if os.path.exists(f"runs/{run_name}"):
            print("Run data for this name already exists - aborting the run to be safe and avoid accidental overwrite")
            sys.exit(1)

        restoring_from_checkpoint = False
        # ensure the directory exists so checkpoints and tensorboard logs can be written
        os.makedirs(os.path.join("runs", run_name), exist_ok=True)
        # TODO use model.cuda(cuda0) rather than initialising every parameter manually
        model = init_model(device)
        # The optimizer isn't aware of the loss function at all - the loss updates tensor gradients by itself,
        # the optimizer then steps in and updates parameters appropriately (taking into account things like
        # learning rate, momentum, decay etc.)
        optimizer = init_optimizer(model)
        scheduler = init_scheduler(optimizer)
        start_epoch = 0
        step = 0
        epoch_size = None  # unknown

    # load training data

    MASK_ID = 103
    if MASK_ID not in tokenizer("[MASK]")["input_ids"]:
        raise Exception("Unknown [MASK] token ID")

    PAD_ID = 0
    if (last := tokenizer("foo", padding="max_length", max_length=6)["input_ids"][-1]) != PAD_ID:
        raise Exception(f"Unknown [PAD] token ID, expected {PAD_ID}, got {last}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    tb_writer = SummaryWriter(os.path.join("runs", run_name))

    for epoch in range(start_epoch, NUM_EPOCHS):
        if stop_requested.is_set():
            break

        print(f"Epoch: {epoch}")

        # if epoch_size is not None then we must have restored from checkpoint so current step is probably
        # not the first of the epoch (and we already know epoch size anyway)
        if epoch == 1 and epoch_size is None:
            epoch_size = step

        if not restoring_from_checkpoint:
            data_it = iter(dataloader)

        # now we start looping, so behave as normal from here on (importantly, initialise data_it afresh for
        # each subsequent epoch)
        restoring_from_checkpoint = False

        for text_batch in data_it:
            optimizer.zero_grad()
            # TODO slide a window over long input sequences?
            seq_len_curriculum(step)
            batch = tokenizer(
                text_batch,
                padding="max_length",  # pad everything to the length indicated by max_length argument
                truncation=True,
                max_length=seq_len_curriculum(step),
                return_tensors="pt",  # return pytorch tensors rather than python lists (another option is 'np' for numpy)
            )["input_ids"]

            batch = batch.to(device)

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

            # TODO try some regularization, I'm seeing 2e+4 magnitude output (total vector, not individual weights)
            loss = loss_fn(y[mask], batch_unmasked[mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            # increment step after optimizer and scheduler updates so step reflects completed batches
            step += 1

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

            if step % 1000 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, epoch_size, step, run_name)

            if stop_requested.is_set():
                save_checkpoint(model, optimizer, scheduler, epoch, epoch_size, step, run_name)
                print("Interrupted. Exiting.")
                break

    tb_writer.close()
