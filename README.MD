# Natural Language Processing with Hugging Face Transformers

*Generative AI Guided Project on Cognitive Class by IBM*

[![Python](https://img.shields.io/badge/PYTHON-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PYTORCH-red?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Name : Vincent Tanjaya**

## My todo :

### 1. Example 1 - Sentiment Analysis

```python
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")

[{'label': 'POSITIVE', 'score': 0.9959210157394409}]

