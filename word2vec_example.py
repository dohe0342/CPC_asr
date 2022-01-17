import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

word2vec.Word2Vec(["happy"], size=5, window=1, negative=3, min_count=1)
