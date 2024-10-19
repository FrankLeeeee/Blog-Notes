# 🦺 Why do we need to use Hugging Face's safetensors?

## 📍 Table of Contents

- [💡 Introduction](#-introduction)
- [🚀 Usage](#-usage)

## 💡 Introduction

This repository contains a simple example of injecting malicious code into your program via pickle and how we can eradicate this risk by using Hugging Face's safetensors.

## 🚀 Usage

```bash
# inject malicious code into all reduce
python hack.py --hack all_reduce

# protect all reduce by using safetensors
python hack.py --hack all_reduce --use-safetensor

# inject malicious code so that your program will be killed after 5 seconds
python hack.py --hack auto_shutdown

# prevent malicious auto_shutdown by using safetensors
python hack.py --hack auto_shutdown --use-safetensor
```
