# Natural Language Processing with Hugging Face Transformers

*Generative AI Guided Project on Cognitive Class by IBM*

[![Python](https://img.shields.io/badge/PYTHON-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PYTORCH-red?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Name : Vincent Tanjaya**

## My todo :

### 1. Example 1 - Sentiment Analysis

```python
# TODO :
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
data = "Artificial intelligence and automation are already causing friction in the workforce. Should schools revamp existing programs for topics like #AI, or are new research areas required?"
specific_model(data)
```

## Result : 
```python
[{'label': 'LABEL_1', 'score': 0.5272253155708313}]
```
Based on the result, model classifies the tweet as neutral or 1, which it actually might be, versus the general model, classifies the same tweet as 'NEGATIVE', which is probably not entirely negative

