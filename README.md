# پروژه ساخت TTS فارسی (Text-to-Speech)

این راهنما یک مسیر **استاندارد و عملی** برای ساخت یک سیستم **تبدیل متن به گفتار (TTS)** فارسی از صفر تا صد با پایتون ارائه می‌دهد. مدل به‌صورت فایل `.h5` ذخیره می‌شود تا بتوانید بعداً دوباره بارگذاری و آموزش آن را ادامه دهید.

> ⚠️ توجه: ساختن یک TTS "بی‌نقص" نیازمند داده زیاد، GPU قوی و معماری‌های پیشرفته (مثل Tacotron2/FastSpeech2 + HiFi-GAN) است. این پروژه پایه‌ای است اما استاندارد و قابل ارتقا.

---

## پیش‌نیازها

### نصب کتابخانه‌ها

```bash
pip install numpy pandas librosa soundfile tensorflow==2.12 tqdm unidecode
```

* **TensorFlow**: برای تعریف و آموزش مدل.
* **Librosa**: پردازش صوت (mel spectrogram).
* **Soundfile**: ذخیره فایل WAV.
* **Unidecode / Regex**: پاک‌سازی متن.
* **TQDM**: نوار پیشرفت.

اگر GPU داری، بهتره نسخه GPU TensorFlow نصب کنی.

---

## ساختار پوشه‌ها

```
tts_project/
├─ data/
│  ├─ metadata.csv        # هر ردیف: "wav_filename|transcript"
│  ├─ wavs/
│  │  ├─ 0001.wav
│  │  ├─ 0002.wav
│  │  └─ ...
├─ checkpoints/
├─ outputs/
├─ main.py
└─ requirements.txt
```

### فرمت دیتاست

* **صوت**: فرمت WAV، مونو، 22050Hz، 16-bit.
* **متن**: فارسی نرمال‌شده، بدون کاراکترهای اضافی.
* **metadata.csv**:

```csv
wavs/0001.wav|سلام به دنیا
wavs/0002.wav|من اسمم سلمان است
```

> 📌 برای شروع، حداقل ۲–۱۰ ساعت داده گفتار تمیز توصیه می‌شود.

---

## فایل main.py

کد اصلی پروژه در فایل `main.py` قرار دارد. این کد شامل بخش‌های زیر است:

1. **پیش‌پردازش متن**: نرمال‌سازی و تبدیل به توکن‌ها.
2. **پیش‌پردازش صوت**: تبدیل WAV به Mel-Spectrogram.
3. **ساخت دیتاست**: بارگذاری همزمان متن و mel.
4. **تعریف مدل**: Encoder-Decoder ساده با Keras.
5. **آموزش و ذخیره مدل به فرمت `.h5`.**
6. **سنتز گفتار**: تولید WAV از متن.
7. **ادامهٔ آموزش**: بارگذاری مدل و ادامه دادن.

---

## آموزش مدل

برای آموزش:

```bash
python main.py --mode train
```

* خروجی: فایل `tts_model.h5` در پوشهٔ اصلی ذخیره می‌شود.
* بهترین checkpoint هم در پوشهٔ `checkpoints/` ذخیره می‌شود.

---

## تست و سنتز گفتار

برای تولید صوت از متن:

```bash
python main.py --mode synth --text "سلام دنیا" --model tts_model.h5
```

خروجی در پوشهٔ `outputs/sample.wav` ذخیره می‌شود.

---

## ادامهٔ آموزش مدل

برای ادامه آموزش از checkpoint:

```bash
python main.py --mode continue --model checkpoints/tts_best.h5
```

مدل جدید به نام `tts_model_cont.h5` ذخیره می‌شود.

---

## دیتاست و آماده‌سازی

### صوت

* ضبط یا جمع‌آوری فایل‌های WAV.
* trim سکوت‌های اضافی.
* نرمال‌سازی سطح صدا.
* تک‌گوینده بهتر از چندگوینده است (کیفیت بالا).

### متن

* نرمال‌سازی: حذف علائم، تبدیل اعداد به حروف.
* (اختیاری) **فونم‌سازی** برای بهبود تلفظ.

---

## عیب‌یابی و مشکلات رایج

* 🔊 **صدا خش‌دار** → مشکل vocoder (Griffin-Lim ساده است → ارتقا به HiFi-GAN).
* 🗣️ **تلفظ اشتباه** → نیاز به فونم‌سازی یا لغت‌نامه.
* 🐌 **کندی آموزش** → کاهش طول sequence یا batch size.
* 📉 **loss کم نمی‌شود** → چک کردن دیتاست و learning rate.

---

## متریک‌ها و ارزیابی

* **MOS (Mean Opinion Score)**: بهترین روش (ارزیابی انسانی).
* **MSE روی mel**: معیار آموزش.
* **MCD (Mel Cepstral Distortion)**: ارزیابی کمی کیفیت صوت.

---

## مسیر ارتقا

1. جایگزین کردن مدل ساده با **Tacotron2** یا **FastSpeech2**.
2. جایگزینی Griffin-Lim با vocoderهای **HiFi-GAN / WaveGlow / WaveRNN**.
3. افزودن **فونم‌سازی فارسی** برای تلفظ دقیق‌تر.
4. جمع‌آوری دیتاست بزرگ‌تر (۱۰–۵۰ ساعت).

---

## بارگذاری مدل برای استفاده مجدد

### بارگذاری مدل ذخیره‌شده:

```python
from tensorflow.keras.models import load_model
model = load_model("tts_model.h5")
```

### ادامهٔ آموزش:

```python
model.fit(dataset, epochs=10)
model.save("tts_model_updated.h5")
```

---

## سخت‌افزار مورد نیاز

* **GPU** پیشنهاد می‌شود (مثلاً RTX 2060 به بالا).
* روی CPU آموزش بسیار کند خواهد بود.
* برای کیفیت بالا و دیتاست بزرگ، نیاز به چند GPU و زمان طولانی دارید.

---

## چک‌لیست سریع (Step by Step)

1. ساختار پوشه را بساز.
2. فایل‌های WAV و metadata.csv را آماده کن.
3. `main.py` را اجرا کن با `--mode train`.
4. بعد از آموزش، با `--mode synth` متن را به گفتار تبدیل کن.
5. برای ارتقا کیفیت → فونم‌سازی + vocoder قوی‌تر.

---

## منابع پیشنهادی

* [Tacotron2 Paper](https://arxiv.org/abs/1712.05884)
* [FastSpeech2 Paper](https://arxiv.org/abs/2006.04558)
* [HiFi-GAN Vocoder](https://arxiv.org/abs/2010.05646)
* [TensorFlowTTS Library](https://github.com/TensorSpeech/TensorFlowTTS)
* [Coqui TTS](https://github.com/coqui-ai/TTS)

---

🎯 با این راهنما، می‌توانی یک سیستم پایهٔ TTS فارسی بسازی، آن را تست کنی و کم‌کم ارتقا بدهی.
