from preprocess import phrase_to_sentences,phrase_to_wordlist
from genModel import genModel
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
import numpy as np 

#load the training and test data
trainData=pd.read_csv("..//data//train.tsv",header=0,delimiter='\t',quoting=3)
testData=pd.read_csv("..//data//test.tsv",header=0,delimiter='\t',quoting=3)

#processing our data
sentences=[]
print("Parsing sentences from training set...")
for phrase in trainData["Phrase"]:
	sentences+=phrase_to_sentences(phrase,remove_stopwords=True)

#generating our word2vec model for our training data
genModel(sentences)

model=Word2Vec.load("..//model//300features_15minwords_10context.w2v")
start=time.time()
#choosing ten words per cluster
word_vectors=model.wv.syn0
num_clusters=int(word_vectors.shape[0]/10)
kmeans_clustering=KMeans(n_clusters=num_clusters)
idx=kmeans_clustering.fit_predict(word_vectors)
end=time.time()
print("Clustering took %d seconds"%(end-start))

#mapping from a word to its cluster assignment
word_centroid_map=dict(zip(model.wv.index2word,idx))

#convert reviews into bag of centroids hence using semantically related clusters
def create_bag_of_centroids(wordlist,word_centroid_map):
    #number of clusters is equal to the highest cluster index
    num_centroids=max(list(word_centroid_map.values()))+1
    bag_of_centroids= np.zeros( num_centroids, dtype="float32" )
    
    #loop over words in review. If word is in the vocabulary find which cluster it belongs to and increment cluster count by 1
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

#create bag of centroids for both training and test set
train_centroids = np.zeros( (trainData["Phrase"].size, num_clusters), dtype="float32" )

clean_train_reviews=[]
for review in trainData["Phrase"]:
    clean_train_reviews.append(phrase_to_wordlist(review,remove_stopwords=True))

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( testData["Phrase"].size, num_clusters),dtype="float32" )

clean_test_reviews = []
for review in testData["Phrase"]:
    clean_test_reviews.append( phrase_to_wordlist( review,remove_stopwords=True ))
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review,word_centroid_map )
    counter += 1

#fit random forest
forest = RandomForestClassifier(n_estimators = 100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,trainData["Sentiment"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"id":testData["PhraseId"], "Sentiment":result})
output.to_csv( "..//predictions//BagOfCentroids.csv", index=False, quoting=3 )