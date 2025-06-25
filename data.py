import torch
from transformers import WhisperFeatureExtractor 
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import random
from utils import (
    download_code_switching_dataset,
    download_eng_dataset,
    concatenate_datasets
)


class DataLoaderLite:
    def __init__(
        self,
        ds,
        batch_size,
        process_rank,
        num_processes,
        shuffle=False,
        collate_fn=None
    ):
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes  
        self.data = ds 
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.reset()

    def __len__(self):
        total_batches = len(self.data) // self.batch_size
        return (total_batches + self.num_processes - 1) // self.num_processes  
    
    def reset(self):
        self.data_indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(self.data_indices)


    def __getitem__(self, idx):
        global_idx = idx * self.num_processes + self.process_rank
        start_idx = global_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))

        if start_idx >= len(self.data):
            raise IndexError("Index out of range")

        batch = []
        for data_idx in self.data_indices[start_idx: end_idx]:
            batch.append(self.data[data_idx])
        
        if end_idx >= len(self.data):
           # if the epoch ends we should reshuffle the data
           self.reset()     
        return self.collate_fn(batch)


class ASRDataset(Dataset):
  def __init__(
      self,
      main_data,
      model_id,
  ):
      self.df = main_data
      self.sample_rate = 16_000
     
      self.feat_ex = WhisperFeatureExtractor.from_pretrained(model_id)


  def __len__(self):
      return len(self.df)

  def __getitem__(self, idx):
     record= self.df[idx]
     wav = torch.from_numpy(record['audio']['array'])
     transcript = record['sentence']
     lang = record['lang']
     mel = self.feat_ex(
        wav,sampling_rate=self.sample_rate, return_tensors='pt', return_attention_mask=True)
     return mel, transcript, lang


def get_dataloader_ddp(
    data,
    model_id,
    process_rank,
    num_processes,
    train_batch_size=64,
    val_batch_size=32,
    test_size=0.1,
    shuffle=True,
    collate_fn=None,
):
    if data == "code":
       train_hf_dataset, test_hf_dataset = download_code_switching_dataset(test_size)
    
    elif data == "en":
        train_hf_dataset, test_hf_dataset= download_eng_dataset(test_size)

    else:
        eng_train, eng_test= download_eng_dataset(test_size)
        code_train, code_test= download_code_switching_dataset(test_size)
        train_hf_dataset = concatenate_datasets([eng_train, code_train])
        test_hf_dataset = concatenate_datasets([eng_test, code_test])

    train_dataset = ASRDataset(train_hf_dataset, model_id)
    test_dataset = ASRDataset(test_hf_dataset, model_id)

    train_loader = DataLoaderLite(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_processes=num_processes,
        process_rank=process_rank,
        collate_fn=collate_fn
    )

    val_loader = DataLoaderLite(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=shuffle,
        num_processes=num_processes,
        process_rank=process_rank,
        collate_fn=collate_fn
    )
    return train_loader, val_loader
