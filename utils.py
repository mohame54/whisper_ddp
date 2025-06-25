import os
import math
import json
import math
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from huggingface_hub.hf_api import HfFolder
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import WhisperTokenizerFast


CODE_SWITCH = "MohamedRashad/arabic-english-code-switching" 


def create_collate_fn(model_id):
    Tok = WhisperTokenizerFast.from_pretrained(model_id, task="transcribe")
    def collate_fn(batch):
        # batch: list of (mel_feat, transcript, lang)
        mels, transcripts = [], []
        for mel, tr, lang in batch:
            mels.append(mel["input_features"].squeeze())
            lang_id = "<ar>" if lang == "ar" else "<en>"
            transcripts.append(lang_id + tr)
        mels = torch.stack(mels)  # [batch, ...]
        data_ids = Tok(transcripts, padding='longest', return_tensors='pt')
        # prepare decoder input_ids and labels
        input_ids = {"decoder_" + k: v[:, :-1] for k, v in data_ids.items()}
        target_ids = {k: v[:, 1:] for k, v in data_ids.items()}
        # ensure decoder_attention_mask exists
        input_ids['decoder_attention_mask'] = torch.ones_like(
            input_ids['decoder_attention_mask'],
            dtype=input_ids['decoder_attention_mask'].dtype
        )
        return mels, input_ids, target_ids
    return collate_fn


def download_code_switching_dataset(test_size=0.1):
    ds = load_dataset(CODE_SWITCH)['train']
    ds = ds.map(lambda x: {"lang":"ar", **x})
    ds = ds.train_test_split(test_size=test_size)
    train = ds['train']
    test = ds['test']
    return train, test

def download_eng_dataset(test_size=0.1):
    ds = load_from_disk("eng_dataset")
    ds = ds.map(lambda x: {"lang":"en", **x})
    ds = ds.train_test_split(test_size=test_size)
    train = ds['train']
    test = ds['test']
    return train, test    


def hf_permission(hf_tok):
    HfFolder.save_token(hf_tok)


def load_json(file_path, as_holder=False):
    with open(file_path, "r") as f:
      data = json.load(f)
    if as_holder:
       data = DataHolder(**data)  
    return data  


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def check_bfloat16_support(logs=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_properties = torch.cuda.get_device_properties(device)

        # Check if the GPU supports bfloat16
        if device_properties.major >= 8:  # Ampere (A100) and newer architectures
            if logs: print(f"GPU {device_properties.name} supports bfloat16.")
            return True
        else:
            if logs: print(f"GPU {device_properties.name} does not support bfloat16.")
    else:
        if logs: print("CUDA is not available on this system.")
    return False


def load_whisper_pretrained(model_id, logs=True, **config):
    dtype = torch.bfloat16 if check_bfloat16_support(logs) else torch.float16
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    return model


class DataHolder:
   def __init__(self, **kwargs):
       for k, v in kwargs.items():
          setattr(self, k, v)

   def __getitem__(self, key):
      return getattr(self, key)


def make_peft_model(
    model,
    logs=True,
    **kwargs
):
    params = dict(
          r=128,
          lora_alpha=32,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
          lora_dropout=0.2,
          bias="none",
      )
    if len(kwargs) == 5:
      params = kwargs
    else:
      params.update(
          kwargs
      )  
    config = LoraConfig(
      **params
    )
    lora_model = get_peft_model(model, config)
    if logs:
      print("Setting Up the lora model with parameters", params)
      print_trainable_parameters(lora_model)
    return  lora_model  



def get_lr_util(it, warmup_steps=200, max_steps=500000, max_lr= 6e-4, min_lr=6e-5):
   # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)       


def cross_loss_fn(logits, input_ids, attention_mask):
    dim = logits.size(-1)
    labels = input_ids.reshape(-1)
    logits = logits.reshape(-1, dim)
    loss = F.cross_entropy(logits, labels, reduction="none")
    attention_mask = attention_mask.to(torch.bool).reshape(-1,)
    loss = loss.masked_select(attention_mask)
    return loss.mean()


def load_json(json_path, env_vars=True):
    json_vars = json.loads(open(json_path, "r").read())
    if env_vars:
        for k in json_vars:
            os.environ.setdefault(k, json_vars[k])
    return json_vars
