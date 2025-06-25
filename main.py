import os
import time
import torch
from huggingface_hub import HfApi
import transformers
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils import (
    hf_permission,
    load_json,
    check_bfloat16_support,
    create_collate_fn,
    load_whisper_pretrained
)
from train_utils import train_epoch, val_epoch
from data import get_dataloader_ddp


assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0
vars = load_json("config.json", env_vars=True)

collate_fn = create_collate_fn(vars['model_id'])

EPOCHS = os.environ['EPOCHS']

EPOCHS_LOGS = os.environ['EPOCHS_LOGS']

ACCUM_STEPS =  os.environ['ACCUM_STEPS']


GLOBAL_TRAIN_BATCH_SIZE =  os.environ['GLOBAL_TRAIN_BATCH_SIZE']
TRAIN_BATCH_SIZE = GLOBAL_TRAIN_BATCH_SIZE // ddp_world_size
VAL_BATCH_SIZE = os.environ['VAL_BATCH_SIZE']


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")



# Load config and HF permissions
config = load_json("config.json")
hf_permission(config.get('hf_perm', {}))
api = HfApi()

# Choose mixed precision: 'bf16' if supported else 'fp16'
dt = "bf16" if check_bfloat16_support() else "fp16"

train_loader, test_loader = get_dataloader_ddp(
    config['data'],
    config['model_id'],
    process_rank=ddp_rank,
    num_processes=4,
    train_batch_size=GLOBAL_TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    collate_fn=collate_fn
)
# Load model
Model = load_whisper_pretrained(config['model_id']).train()
for p in Model.parameters():
    if p.requires_grad:
        p.data = p.data.float()



Opt = torch.optim.AdamW([p for p in Model.parameters() if p.requires_grad], lr=vars['lr'], weight_decay=vars['wd'])

num_training_steps = config['epochs'] * (len(train_loader) + len(test_loader))


Model = DDP(Model, device_ids=[ddp_local_rank])
steps = 0
torch.cuda.empty_cache()
scaler = torch.amp.GradScaler()
for e in range(EPOCHS):
    st = time.time()
    if master_process:
        print(f"Started Training on: {e+1} / {EPOCHS}")
    train_loss = train_epoch(
                    Model,
                    train_loader,
                    Opt,
                    rank=ddp_rank,
                    scaler=scaler,
                    max_norm=1.0,
                    grad_accum_steps=ACCUM_STEPS,
                )
    torch.cuda.empty_cache()
    if master_process:
        val_loss = val_epoch(
                    Model,
                    test_loader,
                    rank = ddp_rank,
                )
        if (e + 1) % config.get('EPOCHS_LOGS', 1) == 0 and ddp_rank == 0:
            checkpoint_path = os.path.join(config['save_dir'], f"{config.get('checkpoint_name', 'checkpoint')}_epoch{e+1}")
            Model.model.save_pretrained(checkpoint_path)
            print(f"Saved model checkpoint: {checkpoint_path}")
            if config.get('push_hf', False):
                api.upload_folder(
                    folder_path=checkpoint_path,
                    path_in_repo=config['hf_repo_path'],
                    repo_id=config['repo_id'],
                )


destroy_process_group()