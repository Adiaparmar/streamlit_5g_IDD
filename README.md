---
title: 5G Intrusion Detection System
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🛡️ 5G Network Intrusion Detection System (CEM-RAG)

An interactive demo for 5G Radio Access Network (RAN) Intrusion Detection using a Two-Tower Contrastive Neural Network and FAISS Episodic Knowledge Retrieval.

## 🚀 Features
- **🎮 Scenario Explorer:** Test pre-loaded real attack traces (DDoS UDP, SYN Flood, ICMP, SSH BruteForce, BGP, etc.) and tune noise/intensity sliders.
- **📝 Manual Input:** Test any custom KPI values or auto-fill from real 24-step attack traces.
- **📡 Live Network Monitor:** Real-time simulated continuous KPI monitoring with on-the-fly attack injection.
- **⚡ Fast CPU Inference:** Complete inference and retrieval runs in under 10 ms on standard CPU.

---

## 💻 Local Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit application
streamlit run app.py
```

---

## ☁️ Deployment Instructions

### Option 1: Streamlit Community Cloud (Easiest & Free)
1. Push this folder to a GitHub repository (e.g. `github.com/<username>/5g-intrusion-detection`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"** -> Choose your repo, branch `main`, and main file `app.py`.
4. Click **Deploy!**

### Option 2: Hugging Face Spaces (Docker)
1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK.
3. Push or upload all files from this directory to the Space repository.
4. Hugging Face will automatically build the container and launch the app on port 7860.
