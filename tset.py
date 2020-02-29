import cloudpickle
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from pytorch_transformers import XLNetTokenizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import pairwise_distances_argmin_min


# nltk.download('punkt')

def get_article_encodings(article_sentences, encoding_tokenizer, encoding_model):
    article_encodings = []
    for sentence in article_sentences:
        input_ids = torch.tensor(encoding_tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
        outputs = encoding_model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        # print(last_hidden_states.shape, type(last_hidden_states), last_hidden_states)
        mean_last_hidden_states = last_hidden_states.mean(-2)
        # print(mean_last_hidden_states.shape, type(mean_last_hidden_states), mean_last_hidden_states)
        article_encodings.append(mean_last_hidden_states.data)
    article_encodings = torch.stack(article_encodings)
    article_encodings = torch.squeeze(article_encodings, dim=1)
    # print(article_encodings.shape, type(article_encodings), article_encodings)
    return cluster_article_encodings(article_sentences, article_encodings)


def cluster_article_encodings(article_sentences, article_encodings, n=3):
    n_clusters = int(np.ceil(len(article_sentences) ** 0.5))
    # print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans = kmeans.fit(article_encodings)
    y_kmeans = kmeans.predict(article_encodings)
    # print(y_kmeans)
    avg = []
    closest = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, article_encodings)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = '\n'.join([article_sentences[closest[idx]] for idx in ordering])
    # print(summary)
    return kmeans, y_kmeans, summary


# previously called get_summary
def get_txt(link):
    with open('txt_scraper.pkl', 'rb') as f:
        func = cloudpickle.load(f)
    txt = func(link)
    return txt


def model(txt):
    tokenized = sent_tokenize(txt)
    xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = torch.load('model')
    model.eval()
    kmeans, y_kmeans, summary = get_article_encodings(tokenized, xlnet_tokenizer, model)
    return summary


joblib.dump(model, 'test_model')
