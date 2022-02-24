import re
import sys
import nltk
import tqdm
import string
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# ===== Customize nltk toolkit =====
stemmer = SnowballStemmer("english")
punctuation_str = string.punctuation
punctuation_set = set(punctuation for punctuation in punctuation_str)
punctuation_set.remove("#")
punctuation_set.remove("!")
stop_words = set(stopwords.words('english'))
stop_words.add('nan')


# ===== Data Analysis =====
def load_data():
    train = pd.read_csv('./datasets/disaster_tweets_train.csv').to_numpy()
    x_train = []
    y_train = []
    for x in train:
        x_train.append(x[-2])
        y_train.append(x[-1])

    return np.array(x_train), np.array(y_train)


def data_analysis():
    x_train, y_train = load_data()
    num_k0 = x_train[y_train == 1].shape[0]  # real disaster
    num_k1 = x_train[y_train == 0].shape[0]  # no real disaster

    print(f'{num_k0 / (num_k0 + num_k1)} tweets talk about real disasters.')
    print(f'{num_k1 / (num_k0 + num_k1)} tweets talk about no real disasters.')


# ===== Train-Test-Split =====
def train_test_split(r):
    x_train, y_train = load_data()
    num_samples = x_train.shape[0]

    indices = list(range(num_samples))
    np.random.shuffle(indices)
    x_temp = x_train[indices]
    y_temp = y_train[indices]

    x_train, x_val = x_temp[:int(r*num_samples)], x_temp[int(r*num_samples):]
    y_train, y_val = y_temp[:int(r*num_samples)], y_temp[int(r*num_samples):]

    return x_train, y_train, x_val, y_val


# ===== Text Preprocessing =====
def process_text(text):
    if isinstance(text, str):
        # Eliminate Hyperlinks form the Tweets: https://stackoverflow.com/a/11332580
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        # Eliminate Retweets
        text = re.sub(r'^RT[\s]+', '', text)
        # Fix Spaces
        text = re.sub(r'%20', ' ', text)
        # Lowercase all the text
        text = text.lower()
        # tokenize text
        word_tokens = nltk.word_tokenize(text)
        # Remove punctuation from the custom punctuation set
        text = [word for word in word_tokens if word not in punctuation_set]
        # Remove stopwords
        text = [word for word in text if word not in stop_words]
        # Lemmatize all words with a PorterStemmer
        text = [stemmer.stem(word) for word in text]
        return ' '.join(word for word in text)
    else:
        return text


# ===== Bag of Words Model =====
def bag_of_words_model(x_train, M, k):

    # build overall word dictionary
    vocab = dict()
    for tweet in tqdm.tqdm(x_train):
        seen_words = []
        for word in tweet.split():
            if word in vocab.keys() and word in seen_words:
                vocab[word]['word_count'] += 1
            elif word in vocab.keys() and word not in seen_words:
                vocab[word]['word_count'] += 1
                vocab[word]['tweet_count'] += 1
                seen_words.append(word)
            else:
                vocab.update({word: {'word_count': 1, 'tweet_count': 1}})

    # incorporate k and M threshold to build bag of words model
    restricted_vocab = dict()
    for word, word_dict in vocab.items():
        if word_dict['tweet_count'] >= k:
            restricted_vocab.update({word: word_dict['word_count']})
    print('Kept ' + str(len(list(restricted_vocab.keys()))) + ' words that occurred in at least ' + str(k) + ' tweets.')
    restricted_sorted_vocab = dict(sorted(restricted_vocab.items(), key=lambda item: item[1]))
    final_vocab_list = list(restricted_sorted_vocab.items())[:M]
    print('Kept ' + str(len(final_vocab_list)) + ' words that occurred most frequently in tweets.')

    return np.array(final_vocab_list)


# ===== Vectorize Data =====
def vectorize_data(data, vocab_array):
    data_vectorized = np.zeros(shape=(data.shape[0], vocab_array.shape[0]))
    for x_idx, x in enumerate(data):
        for word_idx, (word, _) in enumerate(vocab_array):
            if word in x:
                data_vectorized[x_idx, word_idx] = 1

    return data_vectorized


# ===== Prepare Train and Validation Set =====
def preprocess_data(r, M, k):
    x_train, y_train, x_val, y_val = train_test_split(r)

    x_train_processed = []
    for x in x_train:
        x_train_processed.append(process_text(x))
    x_train_processed = np.array(x_train_processed)
    x_val_processed = []
    for x in x_val:
        x_val_processed.append(process_text(x))
    x_val_processed = np.array(x_val_processed)

    vocabs = bag_of_words_model(x_val_processed, M, k)
    x_train_vec = vectorize_data(x_train_processed, vocabs)
    x_val_vec = vectorize_data(x_val_processed, vocabs)

    return x_train_vec, y_train, x_val_vec, y_val


# ===== Build Naive Bayes Classifier =====
class NaiveBayes:

    def __init__(self, X, y, mode='Bernoulli'):
        self.X = X
        self.y = y
        self._mode = mode
        self._classes = np.unique(y)
        self._priors = None
        self._mean = None
        self._var = None

    # compute prior probabilities p(y)
    # compute means and variances for p(x_d|y) for feature d
    def fit(self):
        n_samples, n_features = self.X.shape
        n_classes = len(self._classes)
        self._priors = np.zeros(shape=(n_classes, ), dtype=np.float64)
        self._mean = np.zeros(shape=(n_classes, n_features), dtype=np.float64)
        self._var = np.zeros(shape=(n_classes, n_features), dtype=np.float64)
        for k in self._classes:
            X_k = self.X[self.y == k]
            self._priors[k] = X_k.shape[0] / float(n_classes)
            self._mean[k] = np.mean(X_k, axis=0)
            self._var[k] = np.var(X_k, axis=0)
        self._mean = self._mean.clip(1e-14, 1 - 1e-14)

    def predict(self, X):
        preds = [self._predict(x) for x in X]

        return preds

    def _predict(self, x):
        posteriors = []
        for idx, k in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # p(y)
            class_conditional = self._pdf(x, k)  # log(p(x_1|y)) + log(p(x_d|y)) with d features
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    # get p(x|y) for every feature in x and sum up log values to class conditional
    def _pdf(self, x, k):
        if self._mode == 'Gaussian':
            mean = self._mean[k]
            var = self._var[k]
            numerator = np.exp(- (x-mean)**2 / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            conditional = np.sum(np.log(numerator / denominator))
        elif self._mode == 'Bernoulli':
            mean = self._mean[k]
            conditional = np.sum(x * np.log(mean) + (1-x) * np.log((1 - mean)), axis=0)
        else:
            print('Invalid Probability Distribution.')
            sys.exit(0)
        return conditional


# ===== Calculate Precision, Recall, and F1 Score =====
def calculate_metrics(y_eval, predictions):
    TP_count = 0
    FP_count = 0
    FN_count = 0
    for y_true, y_pred in zip(y_eval, predictions):
        if y_true == 1:
            if y_pred == 1:
                TP_count += 1
            else:
                FN_count += 1
        else:
            if y_pred == 1:
                FP_count += 1
            else:
                continue
    precision = TP_count / (TP_count + FP_count)
    recall = TP_count / (TP_count + FN_count)
    f1_score = 2*(precision*recall) / (precision + recall)

    return precision, recall, f1_score


if __name__ == '__main__':

    # get data information
    data_analysis()

    # get vectorized datasets
    x_train, y_train, x_val, y_val = preprocess_data(r=0.8, M=3000, k=3)

    # Fit Naive Bayes model
    nb = NaiveBayes(X=x_train, y=y_train, mode='Bernoulli')
    nb.fit()
    predictions = nb.predict(x_val)
    nb_precision, nb_recall, nb_f1_score = calculate_metrics(y_val, predictions)
    print("Precision:", nb_precision)
    print("Recall:", nb_recall)
    print("Our Calculated F1 Score:", nb_f1_score)
