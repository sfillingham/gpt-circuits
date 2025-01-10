"""
Training helpers
"""

import inspect
import math
import os
import time
from typing import Optional, Protocol

import torch
import torch.nn as nn
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from config import TrainingConfig
from data.dataloaders import DataLoaderLite


class Trainer(Protocol):
    """
    Base class for a trainer.
    """

    config: TrainingConfig
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: torch.device
    is_master_process: bool
    model: nn.Module
    optimizer: Optimizer
    train_dataloader: DataLoaderLite
    val_dataloader: DataLoaderLite
    best_val_loss: float = float("inf")

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.config = config
        # set up DDP (distributed data parallel).
        # `torchrun` command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        self.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.ddp:
            # use of DDP atm demands CUDA, we set the device appropriately according to rank
            assert torch.cuda.is_available()
            torch.cuda.set_device(self.device)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.ddp_local_rank}")
            self.is_master_process = self.ddp_rank == 0  # this process will do logging, checkpointing etc.
        else:
            # vanilla, non-DDP run
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.is_master_process = True
            self.device = config.device

        # Prepare model
        self.model = model.to(self.device)
        if config.compile:
            self.model = torch.compile(self.model)  # type: ignore

        # Wrap the model if using DDP
        if self.ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])

        # Create optimizer
        self.optimizer = self.configure_optimizer(self.unwrapped_model)

        # Create data loaders
        self.train_dataloader = DataLoaderLite(
            dir_path=config.data_dir,
            B=config.batch_size,
            T=self.unwrapped_model.config.block_size,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split="train",
        )
        self.val_dataloader = DataLoaderLite(
            dir_path=config.data_dir,
            B=config.batch_size,
            T=self.unwrapped_model.config.block_size,
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size,
            split="val",
        )

    def calculate_loss(self, x, y, is_eval: bool) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Returns a tuple of (loss, metrics).
        Metrics are ignored during training but are logged during evaluation.

        :param x: Input tensor.
        :param y: Target tensor.
        :param is_eval: Whether the model is in evaluation mode.
        """
        ...

    def configure_optimizer(self, model: nn.Module) -> Optimizer:
        """
        Configure the optimizer.

        :param model: The model to optimize, which is "unwrapped" using `model.module` if using DDP.
        """
        ...

    @property
    def unwrapped_model(self) -> nn.Module:
        """
        Get the unwrapped model.
        """
        return self.model.module if self.ddp else self.model

    @property
    def autocast_device_type(self) -> str:
        """
        For some reason, autocast doesn't work with "mps", so we fallback to "cpu".
        """
        return "cpu" if self.device.type == "mps" else self.device.type

    @property
    def is_fused_adamW_available(self) -> bool:
        """
        Check if the fused AdamW optimizer is available.
        """
        return "fused" in inspect.signature(torch.optim.AdamW).parameters

    def train(self):
        """
        Train the model.
        """
        # Prepare directory for checkpoints
        if self.is_master_process:
            os.makedirs(self.config.out_dir, exist_ok=True)

        # Set the random seed to make results reproducible.
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)

        # Set the float32 matmul precision to high for better performance.
        torch.set_float32_matmul_precision("high")

        if self.ddp:
            distributed.init_process_group(backend="nccl")

        # Let's see what we're starting with.
        self.val_step(0)

        # Start training.
        for step in range(1, self.config.max_steps + 1):
            self.train_step(step)

            # Always evaluate the model at the end of training.
            last_step = step == self.config.max_steps
            if step % self.config.eval_interval == 0 or last_step:
                self.val_step(step)

        if self.ddp:
            distributed.destroy_process_group()

    def val_step(self, step):
        """
        Perform one step of validation.
        """
        self.model.eval()
        self.val_dataloader.reset()
        with torch.no_grad():
            loss_accum = torch.tensor(0.0, device=self.device)
            for _ in range(self.config.eval_steps):
                x, y = self.val_dataloader.next_batch(self.device)
                with torch.autocast(device_type=self.autocast_device_type, dtype=torch.bfloat16):
                    loss, metrics = self.calculate_loss(x, y, is_eval=True)
                loss_accum += (loss / self.config.eval_steps).detach()

        if self.ddp:
            distributed.all_reduce(loss_accum, op=distributed.ReduceOp.AVG)

        if self.is_master_process:
            loss = loss_accum.item()
            self.log_metrics(
                {
                    "type": "eval",
                    "loss": f"{loss:.4f}",
                    **(metrics or {}),
                }
            )

            # Save the model if it's the best we've seen so far
            self.best_val_loss = min(self.best_val_loss, loss)
            if self.best_val_loss == loss and step > 1:
                print("Saving checkpoint")
                self.unwrapped_model.save(self.config.out_dir)

    def train_step(self, step):
        """
        Perform one step of training optimization.
        """
        t0 = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=self.device)
        for micro_step in range(self.config.gradient_accumulation_steps):
            x, y = self.train_dataloader.next_batch(self.device)
            x, y = x.to(self.device), y.to(self.device)
            if self.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = micro_step == self.config.gradient_accumulation_steps - 1  # type: ignore
            with torch.autocast(device_type=self.autocast_device_type, dtype=torch.bfloat16):
                loss, _ = self.calculate_loss(x, y, is_eval=False)

            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / self.config.gradient_accumulation_steps
            loss_accum += loss.detach()

            loss.backward()

        if self.ddp:
            distributed.all_reduce(loss_accum, op=distributed.ReduceOp.AVG)

        # clip the gradients (if a grad clip value is provided)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip or float("inf"))

        # determine and set the learning rate for this iteration
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        if self.is_master_process and step % self.config.log_interval == 0:
            self.log_metrics(
                {
                    "type": "train",
                    "step": step,
                    "loss": f"{loss_accum.item():.4f}",
                    "lr": f"{lr:.1e}",
                    "norm": f"{norm.item():.4f}",
                    "dt": f"{dt:.3f}",
                }
            )

    def log_metrics(self, metrics: dict):
        """
        Log metrics to the console.
        """
        print(" | ".join([f"{k} {v}" for k, v in metrics.items()]))

    def get_lr(self, step):
        """
        Get the learning rate for a given step. Assumes that step starts at 1 and ends at max_steps.
        """
        # 1) linear warmup for warmup_iters steps
        if step <= self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        # 2) if not decaying, return the learning rate
        if not self.config.decay_lr:
            return self.config.learning_rate
        # 3) if it > max_steps, return min learning rate
        if step >= self.config.max_steps:
            return self.config.min_lr
        # 4) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
