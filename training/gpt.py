"""
Train GPT model:
$ python -m training.gpt --config=shakespeare_128x6

DDP launch for e.g. 8 GPUs:
$ torchrun --standalone --nproc_per_node=8 -m training.gpt --config=shakespeare_128x6
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config.gpt import training_options
from data.dataloaders import DataLoaderLite
from models.gpt import GPT
from training import get_lr

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="shakespeare_64x4", help="Training config")
parser.add_argument("--load_from", type=str, help="Path to load model from")
args = parser.parse_args()

# Load configuration
config_name = args.config
config = training_options[config_name]


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available()
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
    device = config.device
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# output directory for checkpoints and tensorboard logs
if master_process:
    os.makedirs(config.out_dir, exist_ok=True)

# create model
if args.load_from:
    model = GPT.load(args.load_from, device=device)
    print(f"Loaded saved model from {args.load_from}")
else:
    model = GPT(config.gpt_config)
    print("Initialized model from scratch")
model = model.to(device)

if config.compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
unwrapped_model = model.module if ddp else model  # always contains the "raw" unwrapped model

B = config.batch_size  # micro batch size
T = model.config.block_size  # sequence length
grad_accum_steps = config.gradient_accumulation_steps

assert grad_accum_steps % ddp_world_size == 0, "make sure grad_accum_steps is divisible by ddp_world_size"

train_loader = DataLoaderLite(
    dir_path=config.data_dir, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    dir_path=config.data_dir, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)

# optimize!
optimizer = unwrapped_model.configure_optimizers(
    weight_decay=config.weight_decay,
    learning_rate=config.learning_rate,
    device_type=device_type,
    is_master_process=master_process,
)

torch.set_float32_matmul_precision("high")
best_val_loss = float("inf")

for step in range(config.max_steps):
    t0 = time.time()
    last_step = step == config.max_steps - 1

    # once in a while evaluate our validation loss
    if step % config.eval_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = torch.tensor(0.0, device=device)
            val_loss_steps = config.eval_steps
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += (loss / val_loss_steps).detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            val_loss = val_loss_accum.item()
            log_data = {
                "type": "eval",
                "step": step,
                "loss": round(val_loss, 6),
            }
            print(" | ".join([f"{k} {v}" for k, v in log_data.items()]))

            # Save the model if it's the best we've seen so far
            best_val_loss = min(best_val_loss, val_loss)
            if best_val_loss == val_loss and step > 0:
                print("Saving checkpoint")
                unwrapped_model.save(config.out_dir)

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = torch.tensor(0.0, device=device)
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
            _, loss = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip the gradients (if a grad clip value is provided)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip or float("inf"))

    # determine and set the learning rate for this iteration
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process and step % 10 == 0:
        log_data = {
            "type": "train",
            "step": step,
            "loss": round(loss_accum.item(), 6),
            "lr": f"{lr:.4e}",
            "norm": round(norm.item(), 4),
            "dt": round(dt, 4),
            "tok/sec": round(tokens_per_sec, 2),
        }

        print(" | ".join([f"{k} {v}" for k, v in log_data.items()]))

if ddp:
    destroy_process_group()

print(f"Best validation loss: {best_val_loss}")
