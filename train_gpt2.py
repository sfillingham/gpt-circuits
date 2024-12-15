import csv
import math
import os
import time

# -----------------------------------------------------------------------------
import tiktoken
import torch

# run the training loop
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataloaders import DataLoaderLite
from modules.gpt import GPT, GPTConfig, ModelOutput

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

tokens_per_batch = 8192  # 2**19, ~0.5M, in number of tokens
B = 4  # micro batch size
T = model.config.block_size  # sequence length
assert (
    tokens_per_batch % (B * T * ddp_world_size) == 0
), "make sure tokens_per_batch is divisible by B * T * ddp_world_size"
grad_accum_steps = tokens_per_batch // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {tokens_per_batch}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
    dir_path="data/pile_10k",
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
)
val_loader = DataLoaderLite(
    dir_path="data/pile_10k",
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val",
)

torch.set_float32_matmul_precision("high")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    device_type=device_type,
    is_master_process=master_process,
)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
csv_file = open(os.path.join(log_dir, f"{int(time.time())}.csv"), "w")
csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=[
        "type",
        "step",
        "loss",
        "lr",
        "norm",
        "dt",
        "tok/sec",
    ],
)
csv_writer.writeheader()

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_ce_loss_accum = torch.tensor(0.0, device=device)
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model_output: ModelOutput = model(x, y)
                val_ce_loss_accum += (model_output.loss / val_loss_steps).detach()

        if ddp:
            dist.all_reduce(val_ce_loss_accum, op=dist.ReduceOp.AVG)

        val_loss_accum = val_ce_loss_accum

        if master_process:
            log_data = {
                "type": "eval",
                "step": step,
                "loss": round(val_loss_accum.item(), 6),
            }
            csv_writer.writerow(log_data)
            csv_file.flush()
            print(" | ".join([f"{k} {v}" for k, v in log_data.items()]))

            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint_metadata = log_data.copy()
                del checkpoint_metadata["type"]
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    **checkpoint_metadata,
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model_output: ModelOutput = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = model_output.logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    ce_loss_accum = torch.tensor(0.0, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch(device)
        x, y = x.to(device), y.to(device)
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1  # type: ignore
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            model_output: ModelOutput = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        ce_loss = model_output.loss / grad_accum_steps

        ce_loss_accum += ce_loss.detach()

        loss = ce_loss
        loss.backward()

    if ddp:
        dist.all_reduce(ce_loss_accum, op=dist.ReduceOp.AVG)

    loss_accum = ce_loss_accum

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        log_data = {
            "type": "train",
            "step": step,
            "loss": round(loss_accum.item(), 6),
            "lr": f"{lr:.4e}",
            "norm": round(norm.item(), 4),
            "dt": round(dt, 4),
            "tok/sec": round(tokens_per_sec, 2),
        }
        csv_writer.writerow(log_data)
        csv_file.flush()

        print(" | ".join([f"{k} {v}" for k, v in log_data.items()]))

if ddp:
    destroy_process_group()

csv_file.close()
