import time
import csv
import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from progress.bar import Bar
from scipy import spatial
from tqdm import tqdm


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.starttime = 0.0
        self.epoch = 0
        self.tot_epochs = 100
        self.single_epoch_time = 0

    def on_epoch_begin(self, model):
        if self.epoch != 0:
            print(f"Started epoch {self.epoch}.")

    def on_epoch_end(self, model):
        if self.epoch == 0:
            self.starttime = time.time()
            self.single_epoch_time = time.time()
        else:
            if self.epoch != self.tot_epochs:
                print(f"Finished epoch {self.epoch} in {time.time() - self.single_epoch_time}")
                self.single_epoch_time = time.time()
            else:
                print(f"Training finished in {time.time() - self.starttime}s")
        self.epoch += 1


class BiasModel:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')

    def __init__(self, comments_document, comment_column='body', output_name='outputModel',
                 window=4, min_frequency=10, out_dimension=200):
        self.comments_document = comments_document
        self.comment_column = comment_column
        self.output_name = output_name
        self.window = window
        self.min_frequency = min_frequency
        self.out_dimension = out_dimension
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def calculate_centroid(model, words):
        embeddings = [np.array(model[w]) for w in words if w in model]
        centroid = np.zeros(len(embeddings[0]))
        for e in embeddings:
            centroid += e
        return centroid / len(embeddings)

    @staticmethod
    def get_cosine_distance(embedding1, embedding2):
        return spatial.distance.cosine(embedding1, embedding2)

    def load_csv_and_preprocess(self, path, nrowss=None, lemmatise=False):
        """
        input:
        nrowss <int> : number of rows to process, leave None if all
        tolower <True/False> : transform all text to lowercase
        returns:
        List of preprocessed sentences, i.e. the input to train
        """
        print(f"Processing document: {self.comments_document}")
        trpCom = pd.read_csv(self.comments_document, lineterminator='\n', nrows=nrowss)
        trpCom.fillna(0)
        documents = []
        with open(path, 'a', encoding='utf-8') as file:
            for i, row in enumerate(trpCom[self.comment_column]):
                if i % 500000 == 0:
                    print(f'Processing line {i}')
                    for word in documents:
                        file.write("%s\n" % word)
                    documents = []
                try:
                    pp = gensim.utils.simple_preprocess(row)
                    if lemmatise:
                        pp = [wordnet_lemmatizer.lemmatize(w, pos="n") for w in pp]
                    documents.append(pp)
                except TypeError:
                    print(f'Row {i} threw a type error.')
        file.close()
        print(f'Wrote corpus to file: {path}.')
        return documents

    def stream_load_csv_and_preprocess(self, csv_in, csv_out, corpus_out, subsample=False, fraction=None):
        if subsample and fraction is None:
            print("If subsampling is enabled a fraction must be specified.")
            return
        f_in = open(csv_in, encoding="utf-8")
        reader = csv.DictReader(f_in)

        tmp_df = pd.DataFrame()

        previous_day = datetime.fromtimestamp(0)
        with open(corpus_out, 'a', encoding='utf-8') as file:
            for row in tqdm(reader):
                mask = [True if val is not None and val != "" else False for val in row.values()]
                if not all(mask):
                    print("Empty value in row.")
                    continue
                next_day = datetime.fromtimestamp(int(row["created"])).date()

                row["body"] = ' '.join(w for w in gensim.utils.simple_preprocess(row["body"]))

                if previous_day != next_day and not tmp_df.empty:
                    tmp_df["created"] = pd.to_datetime(tmp_df["created"], unit='s').dt.date.astype('datetime64')
                    tmp_df.sort_values(by="created", inplace=True)
                    if subsample:
                        tmp_df = tmp_df.sample(frac=fraction)

                    for comment in tmp_df['body'].tolist():
                        file.write("%s\n" % comment)
                    if subsample:
                        if os.path.isfile(csv_out):
                            tmp_df.to_csv(csv_out, mode='a', header=False, encoding='utf-8', index=False)
                        else:
                            tmp_df.to_csv(csv_out, encoding='utf-8', index=False)

                    tmp_df = pd.DataFrame()

                tmp_df = tmp_df.append(row, ignore_index=True)

                previous_day = next_day

            file.close()

    def train(self, path_to_file, epochs):
        """
        documents list<str> : List of texts preprocessed
        outputfile <str> : final file will be saved in this path
        ndim <int> : embedding dimensions
        window <int> : window when training the model
        minfreq <int> : minimum frequency, words with less freq will be discarded
        epochss <int> : training epochs
        """

        epoch_logger = EpochLogger()
        n_words, n_sentences = self.file_len(path_to_file)
        print(f'Started training model {self.output_name}:\n\tDimensions:{self.out_dimension}, '
              f'\n\tMinimum word frequency:{self.min_frequency} \n\tEpochs:{epochs}')
        model = gensim.models.Word2Vec(corpus_file=path_to_file, size=self.out_dimension, window=self.window,
                                       min_count=self.min_frequency,
                                       workers=5, callbacks=[epoch_logger])
        model.train(corpus_file=path_to_file, total_examples=n_sentences, word_count=n_words, epochs=epochs,
                    total_words=n_words)
        model.save(self.output_name)
        print(f'\nModel saved in {self.output_name}')

    @staticmethod
    def file_len(fname):
        w = []
        with open(fname, encoding="utf-8") as f:
            for i, l in enumerate(f):
                w.append(len(l))
        return np.sum(w), (i + 1)

    def get_top_most_biased_words(self, topk, c1, c2, pos=None):
        """
        topk <int> : topk words
        c1 list<str> : list of words for target set 1
        c2 list<str> : list of words for target set 2
        pos list<str> : List of parts of speech we are interested in analysing
        verbose <bool> : True/False
        """

        if pos is None:
            pos = ['JJ', 'JJR', 'JJS']

        model = Word2Vec.load(self.output_name)
        words_sorted = sorted([(k, v.index, v.count) for (k, v) in model.wv.vocab.items()], key=lambda x: x[1],
                              reverse=False)
        words = [w for w in words_sorted if nltk.pos_tag([w[0]])[0][1] in pos]

        try:
            assert (len(c1) < 1 or len(c2) < 1 or len(words) < 1)
        except AssertionError:
            print('[!] Not enough word concepts to perform the experiment')

        centroid1, centroid2 = self.calculate_centroid(model, c1), self.calculate_centroid(model, c2)
        winfo = []

        print(f"Extracting the {topk} most biased words.")
        bar = Bar('Training', max=len(words))

        for i, w in enumerate(words):
            word = w[0]
            freq = w[2]
            rank = w[1]

            pos = nltk.pos_tag([word])[0][1]

            wv = model[word]

            sent = self.sentiment_analyzer.polarity_scores(word)['compound']

            d1 = self.get_cosine_distance(centroid1, wv)
            d2 = self.get_cosine_distance(centroid2, wv)

            bias = d2 - d1

            winfo.append({'word': word, 'bias': bias, 'freq': freq, 'pos': pos, 'wv': wv, 'rank': rank, 'sent': sent})
            bar.next()

        bar.finish()

        biasc1 = sorted(winfo, key=lambda x: x['bias'], reverse=True)[:min(len(winfo), topk)]
        biasc2 = sorted(winfo, key=lambda x: x['bias'], reverse=False)[:min(len(winfo), topk)]

        for w2 in biasc2:
            w2['bias'] = w2['bias'] * -1

        return [biasc1, biasc2]
