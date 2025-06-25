import torch
from tqdm import tqdm
from utils import cross_loss_fn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


mean = lambda x:sum(x)/len(x)


def train_epoch(
    model,
    train_ds,
    opt,
    rank,
    scaler,
    max_norm=None,
    grad_accum_steps=1,
):
    model.train()
    losses = []
    loop = tqdm(train_ds, desc="Training loop") if rank == 0 else train_ds
    opt.zero_grad()
    loss_accum = 0.0

    for i,mels, inputs, targets in enumerate(loop):
        mels = mels.to(rank)
        inputs = {k:v.to(rank) for k, v in inputs}
        targets = {k:v.to(rank) for k, v in targets}
        if isinstance(model, DDP):
           model.require_backward_grad_sync = (i + 1) % grad_accum_steps == 0
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_features=mels, **inputs)
            logits = outputs.logits
            loss = cross_loss_fn(logits, **targets)
        scaler.scale(loss).backward()
        loss_accum += loss.detach()

        if (i + 1) % grad_accum_steps == 0:
            if max_norm is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            opt.zero_grad()

            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            torch.cuda.empty_cache()
            if rank == 0:
                losses.append(loss_accum.item())
                loop.set_postfix({"Training loss": f"{mean(losses):.4f}"})
            loss_accum = 0.0

    if rank == 0:
        return mean(losses)


@torch.no_grad()
def val_epoch(
    model,
    val_ds,
    rank,
    loss_type="mse_loss",
):
    model.eval()
    loop = tqdm(val_ds, desc="Validation loop") if rank == 0 else val_ds
    losses = []

    for mels, inputs, targets in loop:
        mels = mels.to(rank)
        inputs = {k:v.to(rank) for k, v in inputs}
        targets = {k:v.to(rank) for k, v in targets}
       
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_features=mels, **inputs)
            logits = outputs.logits
            loss = cross_loss_fn(logits, **targets)

        loss = loss.detach().item()

        if rank == 0:
            losses.append(loss)
            loop.set_postfix({"Validation loss": f"{mean(losses):.4f}"})

    if rank == 0:
        return mean(losses)
