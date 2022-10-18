from pickle import bytes_types
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import torch
import torch.optim as optim
import librosa
import json

audio, fs = librosa.load("ori.wav")
with open('vocab.json', encoding='utf-8') as a:
    # 读取文件
    vocab = json.load(a)
target_text = 'THIS IS A TEST'
print(vocab)

def text2id(str,dict):
    ls = []
    for i in str:
        if i == ' ':
            ls.append(dict['|'])
        else:
            ls.append(dict[i])
    return ls

model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a') # Note: PyTorch Model
processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')



# Inference
sample_rate = processor.feature_extractor.sampling_rate
with torch.no_grad():
    model_inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    logits = model(model_inputs.input_values, attention_mask=model_inputs.attention_mask).logits # use .cuda() for GPU acceleration
    pred_ids = torch.argmax(logits, dim=-1).cpu()
    pred_text = processor.batch_decode(pred_ids)
print('Transcription:', pred_text)
model_inputs["labels"] = torch.tensor(text2id(target_text,vocab))

# compute loss
loss = model(**model_inputs).loss
print(loss)

class CWAttack:
    def _init_(self, model, processor, vocab, steps = 1000, lr = 0.01, c = 1):
        self.model = model
        self.processor = processor
        self.vocab = vocab
        self.steps = steps
        self.lr = lr
        self.c = c
    
    def get_distance(mode, x, y):
        if mode == 'l2':
            return torch.mean((x - y) ** 2, dim=1)

    def attack(self, target_text, ori_wav):
        ori_inputs = self.processor(ori_wav, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        adv_inputs = self.processor(ori_wav, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        adv_inputs["labels"] = torch.tensor(text2id(target_text,vocab))
        adv_inputs.requires_grad = True
        best_adv_wav = self.processor(ori_wav, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        best_distance = torch.tensor([1])

        optimizer = optim.Adam([adv_inputs.input_values], lr=self.lr)

        for step in range(self.steps):
            current_distance = self.get_distance('l2', adv_inputs.input_values, ori_inputs.input_values)
            outputs = model(**adv_inputs)
            cost = current_distance + outputs.loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

