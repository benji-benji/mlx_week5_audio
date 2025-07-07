import datasets

ds = datasets.load_dataset("MLCommons/ml_spoken_words", "tt_wav", trust_remote_code=True)
print (ds['train'][0])