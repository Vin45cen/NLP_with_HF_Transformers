# Natural Language Processing with Hugging Face Transformers

*Generative AI Guided Project on Cognitive Class by IBM*

[![Python](https://img.shields.io/badge/PYTHON-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PYTORCH-red?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Name : Vincent Tanjaya**

## My todo :

### 1. Example 1 - Sentiment Analysis

```python
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
data = "Artificial intelligence and automation are already causing friction in the workforce. Should schools revamp existing programs for topics like #AI, or are new research areas required?"
specific_model(data)
```

## Result : 
```python
[{'label': 'LABEL_1', 'score': 0.5272253155708313}]
```

```python
original_model = pipeline("sentiment-analysis")
data = "Artificial intelligence and automation are already causing friction in the workforce. Should schools revamp existing programs for topics like #AI, or are new research areas required?"
original_model(data)
```

## Result : 
```python
[{'label': 'LABEL_1', 'score': 0.5272253155708313}]
```


Based on the result, the model classifies the tweet as neutral or 1, which it actually might be, versus the general model, classifies the same tweet as 'NEGATIVE', which is probably not entirely negative


### 2. Example 2 - Topic Classification

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "I love travelling and learning new cultures",
    candidate_labels=["art", "education", "travel"],
)
```

## Result : 
```python
{'sequence': 'I love travelling and learning new cultures',
 'labels': ['travel', 'education', 'art'],
 'scores': [0.9902300238609314, 0.005778131075203419, 0.003991869743913412]}
```
Based on the result of the given text, the topic is more into travel and it's reasonable since the text also talks about traveling.


### 3. Example 3 - Text Generation Models

```python
generator = pipeline('text-generation', model = 'gpt2')
generator("Hello, I'm a language model", max_length = 30, num_return_sequences=3)
```

## Result : 
```python
[{'generated_text': "Hello, I'm a language model system expert and writing this post was to help you with the syntax.\n\nLet's review,\n\nEach"},
 {'generated_text': "Hello, I'm a language modeler – learning how to program is easy. Why not play music?\n\nThat's a valid question. I"},
 {'generated_text': "Hello, I'm a language model, but I'm not a program writer.\n\nWhat I should know are examples of why this is wrong."}]
```
The text generation model gives a coherent and imaginative of language model, based on the result model performs very well for generating text


### 4. Example 4 - Name Entity Recognition

```python
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
example = "Her name is Anjela and she lives in Seoul."

ner_results = nlp(example)
print(ner_results)
```

## Result : 
```python
[{'entity_group': 'PER', 'score': np.float32(0.9481442), 'word': 'Anjela', 'start': 11, 'end': 18}, {'entity_group': 'LOC', 'score': np.float32(0.9986114), 'word': 'Seoul', 'start': 35, 'end': 41}]
```
based on the result, the model perform well to doing entity recognition. the grouped outputs also relevant and accurate


### 5. Example 5 - Question Answering 

```python
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="Which lake is one of the five Great Lakes of North America?",
    context="Lake Ontario is one of the five Great Lakes of North America. It is surrounded on the north, west, and southwest by the Canadian province of Ontario, and on the south and east by the U.S. state of New York, whose water boundaries, along the international border, meet in the middle of the lake.",
)
```

## Result : 
```python
{'score': 0.9834363460540771, 'start': 0, 'end': 12, 'answer': 'Lake Ontario'}
```
based on the result, the model performs well with distillbert where with the giving question and context it predict the answer is Lake Ontario


### 6. Example 6 - Text Summarization 

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
Lake Superior in central North America is the largest freshwater lake in the world by surface area and the third-largest by volume, holding 10% of the world's surface fresh water. The northern and westernmost of the Great Lakes of North America, it straddles the Canada–United States border with the province of Ontario to the north, and the states of Minnesota to the northwest and Wisconsin and Michigan to the south. It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean.
"""
)
```

## Result : 
```python
[{'summary_text': " Lake Superior is the largest freshwater lake in the world by surface area . It holds 10% of the world's surface fresh water . It straddles the Canada–U.S. border with the province of Ontario to the north . It drains into Lake Huron via St."}]
```
based on the result model can extract the text and summarize it pretty well by using distilbert


### 7. Example 7 - Translation

```python
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Halo, siapa nama anda?")
```

## Result : 
```python
[{'translation_text': 'Bonjour, quel est votre nom ?'}]
```

based on the result, the model succeeded in translating the text from Indonesia to France
