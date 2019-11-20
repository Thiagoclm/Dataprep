# Funções de interesse para se colocar no modulo
# - Letras minusculas
# - mudar type das colunas para diminuir tamanho da base
# - detectar outliers
# - basico pra texto
# - gráficos (boxplots, distribuição, etc...)
# - missings
# - encodings, onehot
# - standardization
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def lower_col(df):
	return map(str(df.columns.lower))

def standard(df):
	scaler = preprocessing.StandardScaler()
	standardized = scaler.fit_transform(df)
	return standardized

def scale(df):
	minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
	x_scale = minmax_scale.fit_transform(x)
	return x_scale

def norm(norm,X):
	normalizer = Normalizer(norm='norm')
	normalizer.transform(X)
	return normalizer

def IQR_outlier(df, min, max):
	Q1 = boston_df_o1.quantile(int(min))
	Q3 = boston_df_o1.quantile(int(max))
	IQR = Q3 - Q1

	df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
	return df


def pca(df):
	pca = PCA()
	x_pca = pca.fit_transform(df)
	x_pca = pd.DataFrame(x_pca)

def mem_usage(pandas_obj):
	# convert float, uint, and category using astype

def ks():

def drop_dup_col(df):
	# drop duplicate features
	df = df.T.drop_duplicates()
	return df

def drop_dup_row(df):
	# drop duplicate rows
	df = df.drop_duplicates()
	return df

def label_enc():


def onehot_enc():
	# get_dummies

def datetime(df, col):
	# convert columns do datetime type and return year, month and day
	df['col'] = pd.to_datetime(df[str(col)])
	df['ano'] = df['col'].dt.year
	df['mes'] = df['col'].dt.month
	df['dia'] = df['col'].dt.day
	df['anomes'] = str(df['ano'])+str(df['mes'])
	return df

def freq_enc(df, col):
	encoding = df.groupby('col').size()
	encoding = encoding/len(df)
	df['col'+'enc'] = df.col.map(encoding)
	return df

def num_ratings():
	# trocar ratings AA-H por nºs

def num_fxatr():
	# troca fx de atraso por nºs

def missings():
	# tratamento de missings

def drop_corr(df, rate):
	# drop highly corrrelated features
	corr_matrix = df.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	to_drop = [column for column in upper.columns if any(upper[column] > float(rate))]
	df.drop(df[to_drop], axis=1, inplace=True)

############## DataViz ##############
def histograms():

def scatter_matrix(df):
	pd.scatter_matrix(df, alpha=0.4)
	plt.show()

def scatterplot():

def corrplot():

def boxplots():

def violinplots():


############## Preprocessing text ##############
def stem(lista):
	# stemming
	pst = PorterStemmer()
	for word in lista :
   		print("word: ", pst.stem(word))


def lemm(lista):
	# lemmatization
	lemmatizer = WordNetLemmatizer() 
	for word in lista:
		print("word: ", lemmatizer.lemmatize(word))

def stopwords(lista, stop_words):
	# stopwords
	# ver codigo do Leandro
	return [word for word in lista if word not in stop_words]

def punctuation(lista):
	# removing punctuation
	 	return lista.translate(str.maketrans('', '', string.punctuation))


# def pos():
# 	# part of speech

def bow(lista):
	# bag of words
	count = CountVectorizer()
	bag_of_words = count.fit_transform(lista)
	feature_names = count.get_feature_names()
	return pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

def tf_idf(df, col):
	# tfidf
	tfidf = TfidfVectorizer()
	response = tfidf.fit_transform(df[str(col)])
	return pd.DataFrame(response.toarray(), columns=tfidf.get_feature_names()) 

def n_grams(df, n):
	# n_grams
    df = df.lower()
    df = re.sub(r'[^a-zA-Z0-9\s]', ' ', df)    
    tokens = [token for token in s.split(" ") if token != ""]    
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

