import torch
import datasets
import torchaudio

languages = ["en_wav", "fr_wav", "de_wav"]  # Add more as needed
sample_per_lang = 100  # Adjust as needed

dss = []
for lang in languages:
    print(f"Loading {lang}...")
    ds = datasets.load_dataset("MLCommons/ml_spoken_words", lang, split="train[:{}]".format(sample_per_lang), trust_remote_code=True)
    ds = ds.add_column("language", [lang] * len(ds))  # Tag language
    dss.append(ds)

ds = datasets.concatenate_datasets(dss)

print (ds['train'][0])

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16_000,  # dataset WAV files are resampled to 16kHz
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

def audio_to_mel(input):
    waveform = torch.from_numpy(input["audio"]["array"])  # shape: [n_samples]
    mel = mel_transform(waveform)  # result: [n_mels, time]
    input["mel_spec"] = mel  # store it in dataset
    return input

ds = ds.map(audio_to_mel, remove_columns=["file", "audio", "speaker_id", "gender", "is_valid", "language"])
print (ds['train'][0])
