from speechbrain.inference.classifiers import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/urbansound8k_ecapa",
    savedir="pretrained_models/gurbansound8k_ecapa",
)

out_prob, score, index, text_lab = classifier.classify_file(
    "speechbrain/urbansound8k_ecapa/dog_bark.wav"
)

print(out_prob, score, index, text_lab)



# load data
# preprocess data
# check shape
# divide into train and test
# set up dataloader
