#using python implementation of word2vec 
from gensim.models import word2vec
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def genModel(sentences,workers=4,size=300,min_count=15,window=10,sample=1e-3):
	print("Training model...")
	model=word2vec.Word2Vec(sentences,workers=workers,size=size,min_count=min_count,window=window,sample=sample)

	#make model memory efficient
	model.init_sims(replace=True)
	model_name="..//model//300features_15minwords_10context.w2v"
	model.save(model_name)