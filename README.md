# nlp-app-recommendation

![](https://i.ibb.co/LNmnk58/1-h-Dq-C5-H28uy-S3-Ic8m-Bm-Sul-Q.png)

### Neuro-linguistic programming (NLP) 

is a psychological approach that involves analyzing strategies used by successful individuals and applying them to reach a personal goal. It relates thoughts, language, and patterns of behavior learned through experience to specific outcomes. 

### HOW NEURO-LINGUISTIC PROGRAMMING WORKS

Modeling, action, and effective communication are key elements of neuro-linguistic programming. The belief is that if an individual can understand how another person accomplishes a task, the process may be copied and communicated to others so they too can accomplish the task. 

Proponents of neuro-linguistic programming propose that everyone has a personal map of reality. Those who practice NLP analyze their own and other perspectives to create a systematic overview of one situation. By understanding a range of perspectives, the NLP user gains information. Advocates of this school of thought believe the senses are vital for processing available information and that the body and mind influence each other. Neuro-linguistic programming is an experiential approach. Therefore, if a person wants to understand an action, they must perform that same action to learn from the experience.

Ref. https://www.goodtherapy.org/learn-about-therapy/types/neuro-linguistic-programming

## Train model

![](https://i.ibb.co/KXY6D4k/NLP.jpg)

```python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(apps["text"].values.astype('U'))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

```
## Predict model

```python 

def get_app_review(app_name):
    index = 0
    
    for word in apps['App']:
        if word == app_name:
            return [index]
        
        index = index + 1


def get_recommendations(app_name):
    id = get_app_review(app_name)
    sim_scores = []
    for idx in id:
        sim_scores = sim_scores + list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        app_indices = [i[0] for i in sim_scores]
    return titles.iloc[app_indices][len(id):]

```

