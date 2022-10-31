# adversarial_asr_attack_wav2vec
This is a code using CW attack against wac2vec 2.0 model (https://huggingface.co/yongjian/wav2vec2-large-a), and is still to be updated.

If you want to run this, first install pytorch : https://pytorch.org/
and transformers : https://huggingface.co/docs/transformers/installation

e.g., installation by conda (or just pip)

```python
conda create -n wav2vec2 python=3.8
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install -c conda-forge transformers

```

After installation, just run the attack file.
```python
python cwattack.py

```

Change the parameters and filename in 'cwattack.py' to create adversarial examples as you like.
