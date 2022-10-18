from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import torch
import librosa
import json
import torch.optim as optim

audio, fs = librosa.load("ori.wav")
adv_audio, adv_fs = librosa.load("adv.wav")
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
print('Transcription:', pred_text[0])
model_inputs["labels"] = torch.tensor(text2id(target_text,vocab))

adv_inputs = processor(adv_audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
disloss = torch.mean((model_inputs.input_values - adv_inputs.input_values) ** 2, dim=1)


# compute loss
ctcloss = model(**model_inputs).loss

loss = disloss + 0.05 * ctcloss

#begin test whether the outputs would change along with optimization

test_inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
test_inputs["labels"] = torch.tensor(text2id(target_text,vocab))

test_inputs.input_values.requires_grad = True
optimizer = optim.Adam([test_inputs.input_values], lr=0.01)
outputs = model(**test_inputs)
cost = outputs.loss
print(outputs)
print(model_inputs.input_values)
print('original w:',test_inputs.input_values)
# do the optimization
for steps in range(5):
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    outputs = model(**test_inputs)
    distance = torch.mean((model_inputs.input_values - test_inputs.input_values) ** 2, dim=1)
    cost = distance + outputs.loss
    print(cost.item())

print(outputs)
if (model_inputs.input_values.equal(test_inputs.input_values)):
    print('fail!')
else:
    print('success!')
    print('later w:',test_inputs.input_values)
    outputs = model(**test_inputs)
    print(outputs)

#class CWAttack:
    #def _init_(self, model, processor, steps = 1000, lr = 0.01, c = 1):

    #def attack(self, target_text, ori_wav):
