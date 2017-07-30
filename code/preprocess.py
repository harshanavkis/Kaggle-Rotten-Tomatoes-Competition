import pandas as pd 

#preprocessing libraries
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import nltk.data

def phrase_to_wordlist(phrase,remove_stopwords=False):
	#remove markup and links
	phrase=BeautifulSoup(phrase).get_text()
	#remove any character other than letters
	phrase=re.sub('[â-zA-Z]'," ",phrase)
	#convert to lower case
	phrase=phrase.lower().split()
	#remove stopwords
	if(remove_stopwords):
		#convert the stopword list into a set for fast parsing
		stops=set(stopwords.words('english'))
		phrase=[word for word in phrase if not word in words]
	#return 
	return phrase

def phrase_to_sentences(phrase,remove_stopwords=False):
	#split phrase into list of sentences containing a list of words
	#nltk tokenizer to split
	tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
	raw_sentence=tokenizer.tokenizer(phrase.strip())

	sentences=[]
	#cleaning the phrases
	for sentence in raw_sentence:
		if(len(sentence)>0):
			sentences.append(phrase_to_wordlist(sentence,remove_stopwords))
	return sentences