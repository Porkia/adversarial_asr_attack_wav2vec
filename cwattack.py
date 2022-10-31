#from pickle import bytes_types
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import torch
import torch.optim as optim
#import librosa
import json
#import soundfile
import wave

audio_file = "ori.wav"
steps = 5000
lr = 4/32768
c = 1
target_text = 'TEXT MY CLIENT HELLO'

# Calculate the SNR.
def SNR_singlech(clean, adv):
    length = min(len(clean), len(adv))
    est_noise = adv[:length] - clean[:length]
    SNR = 10*np.log10((np.sum(clean**2))/(np.sum(est_noise**2)))
    return SNR

# Transform the text into ids according to the vocab dict.
def text2id(str,dict):
    ls = []
    for i in str:
        if i == ' ':
            ls.append(dict['|'])
        else:
            ls.append(dict[i])
    return ls

# Read the audio file as a numpy array, after normalization in [-1,1]
def wav_read(wav_path):
    with wave.open(wav_path, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[0:4]
        strdata = f.readframes(nframes)
        data = np.frombuffer(strdata, dtype=np.int16)
        data = data / 32768
    return data

# Write the audio into a file.
def wav_write(wav_path,data):
    nchannels=1
    sampwidth=2
    framerate=16000
    nframes=len(data)
    comptype='NONE'
    compname='not compressed'
    with wave.open(wav_path, 'wb') as fw:
        fw.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
        data=(data*32768).astype(np.int16)
        fw.writeframes(data.tobytes())

class CWAttack:
    def __init__(self, model, processor, vocab, steps = 1000, lr = 0.01, c = 1):
        self.model = model
        self.processor = processor
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        self.vocab = vocab
        self.steps = steps
        self.lr = lr
        self.c = c
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.rescale = 200/32768    # TODO: finish the tailor function.
    
    # Caculate the distance, currently support the l2 mode.
    def get_distance(self, mode, x, y):
        if mode == 'l2':
            return torch.mean((x - y) ** 2, dim=1)

    def attack(self, target_text, ori_wav):
        ori_inputs = self.processor(ori_wav, sampling_rate=self.sample_rate, return_tensors="pt", padding=True).to(self.device)
        ori_inputs.input_values.requires_grad = False
        adv_inputs = self.processor(ori_wav, sampling_rate=self.sample_rate, return_tensors="pt", padding=True).to(self.device)
        adv_inputs["labels"] = torch.tensor(text2id(target_text,self.vocab))
        adv_inputs.input_values.requires_grad = True
        best_adv_wav = torch.tensor(np.array([ori_wav.copy()])).to(self.device)
        best_distance = torch.tensor([100]).cpu()

        optimizer = optim.Adam([adv_inputs.input_values], lr=self.lr)
        cant_find = 1

        for step in range(self.steps):
            
            current_distance = self.get_distance('l2', adv_inputs.input_values, ori_inputs.input_values)
     
            outputs = self.model(**adv_inputs)
            cost = current_distance + self.c*outputs.loss
            cost = cost.to(self.device)

            pred_ids = torch.argmax(outputs.logits, dim=-1).cpu()
            pred_text = self.processor.batch_decode(pred_ids)
            if (pred_text[0] == target_text):
                self.rescale *= 0.8
                if (current_distance.cpu() < best_distance.cpu()):
                    best_adv_wav = adv_inputs.input_values.clone()
                    best_distance = current_distance.cpu()
                    cant_find = 0
                    print("Get a success example!")
                    self.c = 0.2

            if(step % 100 == 0):
                print("steps ",step, " loss :", cost.item()," text: ", pred_text[0], " current distance: ", current_distance.item(), \
                        "rescale: ", self.rescale)

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            optimizer.step()
            #delta = adv_inputs.input_values - ori_inputs.input_values
            #delta.input_values = delta.clamp(-self.rescale, self.rescale)
            #adv_inputs.input_values = ori_inputs.input_values + delta
           

        if cant_find:
            return [best_adv_wav, adv_inputs.input_values]
        else:
            print("success!")
            return [best_adv_wav, adv_inputs.input_values]

def main():
    # Load the vocab and models
    audio = wav_read(audio_file)

    with open('vocab.json', encoding='utf-8') as a:
        vocab = json.load(a)
    model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a') # Note: PyTorch Model
    processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')
    cwattack = CWAttack(model, processor, vocab, steps, lr, c)
    sample_rate = processor.feature_extractor.sampling_rate
    adv_audio = cwattack.attack(target_text, audio)

    adv1 = adv_audio[0].cpu().detach().numpy()
  

    adv_audio_wav = (adv1[0] * np.sqrt(audio.var() + 1e-7)) + audio.mean()  # Reverse the normalization.
    wav_write('adv_best.wav', adv_audio_wav)

    adv2 = adv_audio[1].cpu().detach().numpy()
    adv2_audio_wav = (adv2[0] * np.sqrt(audio.var() + 1e-7)) + audio.mean()  # Reverse the normalization.
    wav_write('adv_final.wav', adv2_audio_wav)

    # test the adversarial examples.
    adv_audio = wav_read('adv_final.wav')
    print("after read,the adv1 is :", adv_audio)
    with torch.no_grad():
        model_inputs = processor(adv_audio, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(cwattack.device)
        print("after process,the input_values is : ", model_inputs.input_values)
        logits = model(model_inputs.input_values, attention_mask=model_inputs.attention_mask).logits 
        pred_ids = torch.argmax(logits, dim=-1).cpu()
        pred_text = processor.batch_decode(pred_ids)
    print('Transcription of adv:', pred_text[0])
    snr = SNR_singlech(audio, adv_audio)
    print('SNR is : ', snr)

main()
