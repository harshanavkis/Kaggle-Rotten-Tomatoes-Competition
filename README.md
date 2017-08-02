Sentiment Analysis on Movie Reviews
======================================
This repository is my attempt at Kaggle's <a href="https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews">"Sentiment Analysis on Movie Reviews"</a>.

Approach
------------
<li>I created a word2vec representation of my training dataset, with each word being represented as a 300 dimensional vector.</li>
<li>Then I used KMeans clustering to group semantically related words into groups of 10.</li>
<li>Then I converted my training data into a "Bag of Centroids" model and trained a Random Forest Classifier,which was used for predictions on the test set</li>

Dependencies Used
-------------------
<li>Numpy</li>
<li>gensim.Word2Vec</li>
<li>sklearn</li>

Result
-------
My model acheived a score of 0.5404.
