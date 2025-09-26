# main.py
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import unicodedata
import re
from unidecode import unidecode

# --------------------
# تنظیمات (قابل تغییر)
# --------------------
SR = 22050
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
FRAME_SHIFT = HOP_LENGTH
BATCH_SIZE = 16
EPOCHS = 100
MAX_TEXT_LEN = 200

DATA_DIR = "data"
METADATA = os.path.join(DATA_DIR, "metadata.csv")
WAV_DIR = os.path.join(DATA_DIR, "wavs")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
OUTPUTS = "outputs"
os.makedirs(OUTPUTS, exist_ok=True)

# --------------------
# متن: نرمال‌سازی ساده
# --------------------
def normalize_text(txt):
    txt = txt.strip()
    # برداشتن کاراکترهای غیرضروری — میتونی قواعد بیشتری اضافه کنی
    txt = re.sub(r"[«»\"()؟!ـ…:؛،\.\,]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.lower()
    txt = txt.strip()
    return txt

# --------------------
# vocabulary ساده (حروف و space)
# --------------------
# اینجا برای فارسی ساده‌سازی می‌کنیم؛ میتونی فونم یا لِکسیکون اضافه کنی
chars = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءآؤئئ‌ ")  # مثال (اضافه/کم کن)
# اطمینان: include space
vocab = ["<pad>", "<unk>"] + chars
char_to_id = {c:i for i,c in enumerate(vocab)}
id_to_char = {i:c for c,i in char_to_id.items()}

def text_to_sequence(text):
    text = normalize_text(text)
    seq = []
    for ch in text:
        if ch in char_to_id:
            seq.append(char_to_id[ch])
        else:
            seq.append(char_to_id["<unk>"])
    return seq

# --------------------
# محاسبه mel-spectrogram
# --------------------
def wav_to_mel(path):
    y, sr = librosa.load(path, sr=SR)
    # trim silence (اختیاری)
    y, _ = librosa.effects.trim(y)
    # mel spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=N_FFT,
                                       hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                       n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    # normalize to -4..4 or 0..1
    S_norm = (S_db + 80.0) / 80.0  # scale to 0..1 (if db min -80)
    return S_norm.T  # time x n_mels

# --------------------
# دیتاست ساده با generator
# --------------------
class TTSDataset(tf.keras.utils.Sequence):
    def __init__(self, metadata_csv, wav_dir, batch_size=BATCH_SIZE, shuffle=True):
        df = pd.read_csv(metadata_csv, sep="|", header=None, quoting=3)
        df.columns = ["wav", "txt"]
        self.items = df.values.tolist()
        self.wav_dir = wav_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.items))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.items) + self.batch_size - 1)//self.batch_size

    def __getitem__(self, idx):
        batch_items = [self.items[i] for i in self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]]
        texts, mels, text_lens, mel_lens = [], [], [], []
        for wav_rel, txt in batch_items:
            wav_path = os.path.join(self.wav_dir, wav_rel)
            mel = wav_to_mel(wav_path)  # T x n_mels
            seq = text_to_sequence(txt)
            texts.append(seq)
            mels.append(mel)
            text_lens.append(len(seq))
            mel_lens.append(len(mel))
        # padding
        max_text = max(text_lens)
        max_mel = max(mel_lens)
        text_padded = np.zeros((len(texts), max_text), dtype=np.int32)
        mel_padded = np.zeros((len(mels), max_mel, N_MELS), dtype=np.float32)
        for i, seq in enumerate(texts):
            text_padded[i, :len(seq)] = seq
        for i, mel in enumerate(mels):
            mel_padded[i, :mel.shape[0], :] = mel
        return [text_padded, np.array(text_lens), np.array(mel_lens)], mel_padded

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --------------------
# مدل ساده: encoder-decoder-attention
# --------------------
def build_model(vocab_size, embed_dim=256, enc_units=256, dec_units=256):
    # inputs
    text_in = layers.Input(shape=(None,), dtype="int32", name="text_in")
    text_len = layers.Input(shape=(), dtype="int32", name="text_len")
    # encoder
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(text_in)
    x = layers.Conv1D(256, 5, padding="same", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(enc_units, return_sequences=True))(x)  # (B, T_text, 2*enc_units)
    encoder_outputs = layers.Dense(dec_units, activation=None)(x)  # project

    # simple attention mechanism (Luong)
    # decoder will be autoregressive — for simplicity we predict all frames at once using teacher forcing during training.
    # We'll use a simple decoder: take encoder summary and upsample/transform to mel frames via Dense + Conv1DTranspose
    # NOTE: This is a simplified approach (non-autoregressive style) for learning mapping to mel length via Conv/upsampling.
    # For production use Tacotron2/Transformer-TTS.

    # get fixed-length summary
    enc_mean = layers.GlobalAveragePooling1D()(encoder_outputs)
    # expand time dimension (we will predict mel frames with a small upsample factor)
    # predict sequence length dynamically: we will predict same number frames as mel in training using MSE loss
    # so during training we rely on mel targets to set length. For inference we'll use a simple heuristic.
    decoder_hidden = layers.Dense(dec_units, activation="relu")(enc_mean)
    # project to mel features over time: use a small upsampling network
    # We'll output a sequence using a simple RepeatVector followed by Conv1D
    def build_decoder_heads():
        rep = layers.RepeatVector(200)(decoder_hidden)  # 200 frames default, during training we'll trim
        y = layers.Conv1D(512, 3, padding="same", activation="relu")(rep)
        y = layers.Conv1D(256, 3, padding="same", activation="relu")(y)
        y = layers.TimeDistributed(layers.Dense(N_MELS, activation="sigmoid"))(y)  # normalized 0..1
        return y

    mel_out = build_decoder_heads()  # (B, T_pred, N_MELS)

    model = Model(inputs=[text_in, text_len], outputs=mel_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

# --------------------
# Griffin-Lim vocoder (ساده)
# --------------------
def mel_to_audio(mel_norm, n_iter=60):
    # inverse normalization (assuming S_db scaled to 0..1 as above)
    S_db = mel_norm.T * 80.0 - 80.0  # n_mels x T
    S = librosa.db_to_power(S_db)
    # mel basis inversion: approximate STFT magnitude
    inv_mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    # Use librosa's pseudo-inverse via librosa.feature.inverse.mel_to_stft
    S_stft = librosa.feature.inverse.mel_to_stft(S, sr=SR, n_fft=N_FFT, power=1.0)
    # griffin-lim
    y = librosa.griffinlim(S_stft, n_iter=n_iter, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    return y

# --------------------
# آموزشی و تست
# --------------------
def train():
    # dataset
    ds = TTSDataset(METADATA, WAV_DIR, batch_size=BATCH_SIZE)
    vocab_size = len(vocab)
    model = build_model(vocab_size)
    model.summary()
    ckpt = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "tts_best.h5"), save_best_only=True, monitor="loss", mode="min")
    model.fit(ds, epochs=EPOCHS, callbacks=[ckpt])
    # save final
    model.save("tts_model.h5")
    print("Model saved to tts_model.h5")

# --------------------
# بارگذاری مدل .h5 و نمونه inference
# --------------------
def load_and_synthesize(text, model_path="tts_model.h5", out_wav="outputs/sample.wav"):
    # اگر مدل کاستوم لایه داشت، باید custom_objects بدهی؛ اینجا ساده است.
    model = tf.keras.models.load_model(model_path)
    seq = np.array([text_to_sequence(text)])
    text_len = np.array([len(seq[0])])
    # model outputs shape (1, T_pred, N_MELS)
    mel_pred = model.predict([seq, text_len])[0]  # T_pred x N_MELS (normalized 0..1)
    # trim trailing zeros
    mel_pred = np.clip(mel_pred, 0.0, 1.0)
    # inverse to waveform
    wav = mel_to_audio(mel_pred)
    sf.write(out_wav, wav, SR)
    print("WAV written to:", out_wav)

# --------------------
# ادامهٔ آموزش (load checkpoint and fit more)
# --------------------
def continue_training(checkpoint_path="checkpoints/tts_best.h5", extra_epochs=10):
    ds = TTSDataset(METADATA, WAV_DIR, batch_size=BATCH_SIZE)
    model = tf.keras.models.load_model(checkpoint_path)
    ckpt = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "tts_best_cont.h5"), save_best_only=True, monitor="loss", mode="min")
    model.fit(ds, epochs=extra_epochs, callbacks=[ckpt])
    model.save("tts_model_cont.h5")
    print("Continued training and saved to tts_model_cont.h5")

# --------------------
# CLI ساده
# --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","synth","continue"], default="train")
    p.add_argument("--text", default="سلام دنیا")
    p.add_argument("--model", default="tts_model.h5")
    args = p.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "synth":
        load_and_synthesize(args.text, model_path=args.model)
    elif args.mode == "continue":
        continue_training(checkpoint_path=args.model, extra_epochs=10)
