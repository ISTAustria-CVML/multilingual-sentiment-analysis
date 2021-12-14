# Multilingual Sentiment Analysis for Tweets
Code and Pretrained Models for `Overcoming Rare-Language Discriminationin Multi-Lingual Sentiment Analysis` (IEEE BigData Special Session MLDB 2021)

Update: version 2 of the model (more accurate fine-tuned transformer model)
is available via
[https://huggingface.co/clampert/multilingual-sentiment-covid19](https://huggingface.co/clampert/multilingual-sentiment-covid19)


## Model v1

The original sentiment analysis model consists of feature 
extraction followed by a linear classifier. 

```
from sentence_transformers import SentenceTransformer
smodel = SentenceTransformer('stsb-xlm-r-multilingual')
params = np.loadtxt("parameters.txt")
w,b = params[:,-1],params[-1]

vec = self.smodel.encode("I am happy.")
score = np.dot(vec,w)+b
print(score)
```

For a minimal working code example see the `predict-sentiment-v1.py` file.

## Model v2

The improved sentiment analysis model consists of finetuned 
deep network. Using it is extremely simple using Huggingface's
`transformer` package:

```
from transformers import pipeline
classifier = pipeline("text-classification", "clampert/multilingual-sentiment-covid19")

$ classifier("I am happy.")
{'label': 'positive', 'score': 0.918508768081665}

$ classifier("Ich bin traurig!")
{'label': 'negative', 'score': 0.97398442029953}

For a minimal working code example see the `predict-sentiment-v2.py` file.


### Citation

If you use either of the models, please cite our IEEE BigData paper:

```
@inproceedings{lampert2021overcoming,
  title={Overcoming Rare-Language Discrimination in Multi-Lingual Sentiment Analysis},
  author={Jasmin Lampert and Christoph H. Lampert},
  booktitle={IEEE International Conference on Big Data (BigData)},
  year={2021},
  note={Special Session: Machine Learning on Big Data},
}
```
