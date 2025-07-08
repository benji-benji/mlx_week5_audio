import whisper
import torch
import json
import toolz
import random
import datetime
import tqdm
import wandb
import os 

def random_excluding(start, stop, exclude):
    while True:
        n = random.randint(start, stop)
        if n != exclude:
            return n


class ReefSetDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.home = "./corpus/ReefSet_v1.0"
        self.cache_dir = "./cache/ReefSet_v1.0"
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.home + "/reefset_annotations.json") as f: self.metadata = json.load(f)
        self.cats = ["dolphin", "boat", "other"]
        def filter(i):
            for cat in self.cats[:-1]:
                if cat in i['label']: return cat
            return self.cats[-1]
        self.metadata = toolz.groupby(filter, self.metadata)
        self.len = min([len(v) for _,v in self.metadata.items()]) * len(self.cats)
        print ("Dataset loaded:", {(k,len(v)) for k,v in self.metadata.items()})
        
    def set_mel_function(self, f):
        self.mel = f
        
    def get_classes(self):
        return self.metadata.keys()
    
    def get_cat_item(self, cat, idx):
        cache_path = self.cache_dir + "/" + self.metadata[cat][idx]["file_name"] + ".mel.pt"
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        path = self.home + "/full_dataset/" + self.metadata[cat][idx]["file_name"]
        mel = self.mel(path)
        torch.save(mel, cache_path)
        return mel
        
    def get_all_cats_pos_neg_item(self, iidx):
        cat_id = iidx%len(self.cats)
        idx = iidx // len(self.cats) + iidx%len(self.cats)
        cat = self.cats[cat_id]
        negative_cat = random.choice([c for c in self.cats if c != cat])
        anchor = self.get_cat_item(cat, random_excluding(0, len(self.metadata[cat])-1, idx))
        positive = self.get_cat_item(cat, idx)
        negative = self.get_cat_item(negative_cat, random.randint(0, len(self.metadata[negative_cat])-1))
        ret = [anchor, positive, negative, cat_id]
        return ret
            
    def __getitem__(self, idx):
        return self.get_all_cats_pos_neg_item(idx)
    
    def __len__(self):
        return self.len


class WhisperEncoderClassifier(torch.nn.Module):
    def __init__(self, num_classes, whisper_model="tiny"):
        super().__init__()
        # Load Whisper (e.g., tiny for fast embedding tests)
        self.whisper = whisper.load_model(whisper_model, download_root="./cache")
        self.hidden_size = self.whisper.dims.n_audio_state
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )
        self.device = self.whisper.device
        self.to(self.whisper.device)
        
    def forward(self, mel):  # mel: [B, 80, T]
        with torch.no_grad():  # Freeze encoder
            enc = self.whisper.encoder(mel)  # [B, T, hidden_size]
        enc = enc.mean(dim=1)     # [B, hidden_size]
        logits = self.classifier(enc)
        return logits
    
ds = ReefSetDataset()
model = WhisperEncoderClassifier(len(ds.get_classes()))
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)  # only classifier trainable

ds.set_mel_function(lambda f: whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(f))))
loader = torch.utils.data.DataLoader(ds, 128, num_workers=6)

ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
wandb.init(entity="flplv-private", project=f'mli-week-05-dolphins-whisper', name=f"{ts}")

wandb_step = 0
for epoch in range(1000):
    epoch_loader = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}", leave=True)
    
    for _, (anc, pos, neg, labels) in enumerate(epoch_loader):
        model.train()
        
        anc_emb = model(anc.to(model.device))
        pos_emb = model(pos.to(model.device))
        neg_emb = model(neg.to(model.device))
        
        triplet_loss = torch.nn.functional.triplet_margin_loss(
            anc_emb, pos_emb, neg_emb, margin=0.5, p=2
        )
        
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        
        d_ap = torch.nn.functional.pairwise_distance(anc_emb, pos_emb, p=2)
        d_an = torch.nn.functional.pairwise_distance(anc_emb, neg_emb, p=2)

        # Accuracy = % of triplets where anchor is closer to positive
        correct = (d_ap < d_an).float()
        accuracy = correct.mean()
        
        epoch_loader.set_postfix_str(f"loss:{triplet_loss.item():.2f} acc:{accuracy.item():.2f}")
        
        wandb_step += len(labels)
        wandb.log({'loss': triplet_loss.item(), "acc":accuracy.item()}, step=wandb_step)
        if epoch != 0 and (epoch % 5) == 0: torch.save(model.classifier.state_dict(), f'./checkpoints/classifier-{ts}.{epoch}.pth')
        
wandb.finish()
