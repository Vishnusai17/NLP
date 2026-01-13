# Google Colab Training Instructions

## 🎯 Goal
Train a high-quality LSTM model in Google Colab (free GPU!) and use it locally with your web interface.

---

## 📋 Step-by-Step Guide

### 1️⃣ Upload to Google Colab

1. Go to **https://colab.research.google.com/**
2. Click **File → Upload notebook**
3. Upload `colab_training.py` (rename to `colab_training.ipynb` if needed, or create new notebook and copy the code)

### 2️⃣ Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU (T4 or better)**
3. Click **Save**

### 3️⃣ Run Training

1. Click **Runtime → Run all** or press `Ctrl+F9`
2. Wait for training to complete (~20-40 minutes)
3. Watch the progress bars!

**Expected Output:**
```
📊 LOADING IMDB DATASET
Processing 10000 reviews...
✓ Loaded 10000 reviews

🔤 Tokenizing text...
✓ Vocabulary size: 10000

🔢 Creating N-gram sequences...
✓ Created 150000+ training sequences

🧠 BUILDING LSTM MODEL
Model: "sequential"
...

🚀 TRAINING MODEL (GPU Accelerated)
Epoch 1/50
...
```

### 4️⃣ Download Model Files

After training completes, download these **3 files** from Colab:

1. **movie_critic_model.h5** (the trained model)
2. **tokenizer.pickle** (vocabulary mappings)
3. **model_config.pickle** (model configuration)

**How to Download:**
- Click the folder icon 📁 in the left sidebar
- Right-click each file → Download

### 5️⃣ Place Files Locally

Put the 3 downloaded files in your local project folder:

```
LSTM/
├── movie_critic_model.h5       ← Download here
├── tokenizer.pickle             ← Download here
├── model_config.pickle          ← Download here
├── app.py
├── movie_critic_chatbot.py
└── ...
```

### 6️⃣ Run Locally

```bash
python app.py
```

Visit: **http://localhost:5001**

---

## 🎨 The Difference

| Feature | Local Training | Colab Training |
|---------|---------------|----------------|
| **Samples** | 500 reviews | **10,000 reviews** |
| **Vocabulary** | 3,000 words | **10,000 words** |
| **LSTM Size** | 50/32 units | **150/100 units** |
| **Time** | 5-10 min | 20-40 min |
| **Quality** | Basic | **High Quality** ✨ |
| **Memory Issues** | ❌ Killed | ✅ No problem |

---

## ⚙️ Adjusting Settings

If Colab still runs out of memory, edit `colab_training.py`:

```python
# Reduce samples
X, y, total_words = chatbot.load_and_prepare_data(num_samples=5000)

# Or reduce vocabulary
chatbot = MovieCriticChatbot(
    vocab_size=5000,  # Instead of 10000
    ...
)
```

---

## 🎬 Expected Quality

With Colab training, you'll get **much better** text generation:

**Before (Local, 500 samples):**
```
"this movie is <OOV> than this <OOV> trying all a continues..."
```

**After (Colab, 10,000 samples):**
```
"this movie is a masterpiece with brilliant acting and stunning cinematography..."
```

Much more coherent! ✨

---

## ❓ Troubleshooting

**Issue**: "Out of memory" in Colab
**Solution**: Reduce `num_samples` to 5000 or 3000

**Issue**: Training is very slow
**Solution**: Make sure GPU is enabled (Runtime → Change runtime type)

**Issue**: Files won't download
**Solution**: Use `from google.colab import files` then `files.download('movie_critic_model.h5')`

---

## 🚀 You're Done!

Once you place the 3 files locally and run `python app.py`, your web interface will use the **high-quality Colab-trained model**! 🎉
