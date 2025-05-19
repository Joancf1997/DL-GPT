# DL-GPT

A project inspired by the book ***Build a Large Language Model (From Scratch)***, this repository implements a transformer-based LLM with both **text classification** and **text generation** capabilities. It includes a user-friendly UI built with **Vue.js** for seamless interaction with the model.

![Vue UI + LLM](https://github.com/Joancf1997/DL-GPT/raw/main/assets/preview.png)

## 🔍 Features

* 🚀 Transformer architecture implemented from scratch using PyTorch
* 🗣️ **Text generation** (predict the next word or sentence continuation)
* 🏷️ **Text classification** (assign labels to text input)
* 💻 Interactive **Vue.js frontend**
* 📊 Training and evaluation on custom datasets
* 💬 Model inference through the web UI

## 🛠️ Setup Instructions

### 🔧 Backend

1. **Create a Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend server**

   ```bash
   python app.py
   ```

---

### 🌐 Frontend (Vue.js)

1. **Navigate to the UI directory**

   ```bash
   cd ui
   ```

2. **Install frontend dependencies**

   ```bash
   npm install
   ```

3. **Run the Vue app**

   ```bash
   npm run dev
   ```

The UI will be accessible at `http://localhost:5173` (or the port shown).

---

## 🧠 Model Training

To train the model from scratch:

```bash
python trainer/train.py
```

To evaluate:

```bash
python trainer/evaluate.py
```

Modify the configuration or paths as needed inside the `trainer/` folder.

---

## 🧪 Example Use Cases

* **Text Generation**: Enter a text prompt in the UI, and the model will generate a coherent continuation.
* **Text Classification**: Choose from predefined categories (e.g., spam detection, sentiment analysis) and classify your input.

---

## 📚 Inspired By

This project is based on the excellent book:
***Build a Large Language Model (From Scratch)***
By: [Sebastian Raschka](https://sebastianraschka.com)

---
