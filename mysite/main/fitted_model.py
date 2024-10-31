from datasets import load_dataset
import torch
import torch
import random
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score, Precision, Accuracy
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
import pytorch_lightning as pl
import tqdm
import pandas as pd
from torch.nn import functional as F
import re
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub.hf_api import HfFolder
from pytorch_lightning.callbacks import ModelCheckpoint
from huggingface_hub.hf_api import HfFolder
from pytorch_lightning.callbacks import ModelCheckpoint


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
class QAClassificationModel(pl.LightningModule):
    def __init__(self, train_dataloader_size, model_name='bert-base-multilingual-cased', learning_rate=1e-5, dropout_prob=0.3, warmup_steps=1500):
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

        # Инициализируем mBERT
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to('cpu')
        self.model.dropout = torch.nn.Dropout(self.dropout_prob)

        # Настройка LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=512,
            lora_alpha=1024,
            bias="all"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Инициализируем F1-метрику для бинарной классификации
        self.f1_metric = F1Score(num_classes=2, task='binary')
        self.precision_metric = Precision(num_classes=2, task='binary')
        self.accuracy_metric = Accuracy(num_classes=2, task='binary')

        self.train_dataloader_size = train_dataloader_size

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        return self.model(input_ids=input_ids.to('cpu'), attention_mask=attention_mask.to('cpu'), token_type_ids=token_type_ids.to('cpu'), labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            token_type_ids=batch['token_type_ids'], 
            labels=batch['labels']
        ).to('cpu')
        loss = outputs.loss

        preds = torch.argmax(outputs.logits, dim=1).to('cpu')
        labels = batch['labels']
        
        accuracy = self.accuracy_metric(preds, labels)
        self.log('train_accuracy', accuracy, prog_bar=True, logger=True)

        f1 = self.f1_metric(preds, labels)
        self.log('train_f1', f1, prog_bar=True, logger=True)
        
        precision = self.precision_metric(preds, labels)
        self.log('train_precision', precision, prog_bar=True, logger=True)

        # Логирование лосса
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            token_type_ids=batch['token_type_ids'], 
            labels=batch['labels']
        )
        loss = outputs.loss

        # Предсказания и реальные значения для расчета метрики
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
        accuracy = self.accuracy_metric(preds, labels)
        self.log('val_accuracy', accuracy, prog_bar=True, logger=True)

        f1 = self.f1_metric(preds, labels)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        
        precision = self.precision_metric(preds, labels)
        self.log('val_precision', precision, prog_bar=True, logger=True)


        return loss

    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        scheduler = self.scheduler(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

model2 = QAClassificationModel.load_from_checkpoint('main/epoch-epoch=02.ckpt')


def get_result(question, answer):
    input_ = tokenizer(
                question, answer, 
                padding='max_length', 

                truncation=True, 
                max_length=256, 
                return_tensors="pt"
            )
    input_ = {key: val.squeeze(0) for key, val in input_.items()}
    preds = torch.argmax(model2(
        input_ids=input_['input_ids'].reshape(1,-1), 
        token_type_ids=input_['token_type_ids'].reshape(1,-1), 
        attention_mask=input_['attention_mask'].reshape(1,-1)
    ).logits, dim=1)
    return int(preds)
