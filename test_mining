import pandas as pd
import numpy as np
import requests
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import time

start = time.time()

# Read the Corona_NLP_train.csv
df1 = pd.read_csv(r"Corona_NLP_train.csv", sep = ",", engine="python", encoding = "latin-1")

# Compute the possible sentiments that a tweet may have
possible_sentiments = df1["Sentiment"].unique()
print("Possible sentiments that a tweet may have:", possible_sentiments)

# Create a DataFrame named df2
df2 = pd.DataFrame()

# Count the number of each sentiment
df2["Count"] = df1["Sentiment"].value_counts()

# Calculate the percentage of each sentiment
df2["Percentage"] = round(df2["Count"]/len(df1)*100, 2)

# Sort by descending order the percentage of each sentiment
df2 = df2.sort_values(by = "Percentage", ascending=False)

# Locate the second most popular sentiment in the tweets
second_most_popular_sentiment = df2["Percentage"].iloc[[1]]
print("Second most popular sentiment in the tweets:", second_most_popular_sentiment)

# Keep only the rows with the sentiment Extremely Positive
df3 = df1[df1["Sentiment"] == "Extremely Positive"]

# Drop all the unnecessary  columns
df3 = df3.drop(columns = ["UserName", "ScreenName", "Location", "OriginalTweet"])

# Group the number of tweets by date
df4 = df3.groupby("TweetAt").count()

# Sort by descending order the number of tweets by date with the sentiment Extremely Positive
df4 = df4.sort_values(by = "Sentiment", ascending=False)

# Reset the index
df4 = df4.reset_index()

# Locate the date with the greatest number of Extremely Positive tweets
date_greatest_nb_extremely_positive_tweets = df4["TweetAt"].iloc[[0]]
print("Date with the greatest number of extremely positive tweets:", date_greatest_nb_extremely_positive_tweets)

# Convert the column to lower case
df1["OriginalTweet"] = df1["OriginalTweet"].str.lower()

# Remove URLs from the text as they do not help understanding the content of the tweets
df1["OriginalTweet"] = df1["OriginalTweet"].apply(lambda x: re.sub(r"http://\S+|https://\S+", " ", str(x)))

# Replace non-alphabetical characters with whitespaces
df1["OriginalTweet"] = df1["OriginalTweet"].apply(lambda x: re.sub(r"[^a-zA-Z]", " ", str(x)))

# Replace whitespaces by a single whitespace
df1["OriginalTweet"] = df1["OriginalTweet"].apply(lambda x: re.sub(r" +", " ", str(x)))

# Tokenize the tweets
df1["Tokenized"] = df1["OriginalTweet"].apply(lambda x: x.split())

# Pandas Series of all the words in the tweets
original_tweets = pd.Series(np.concatenate(df1["Tokenized"]))

# Count the number of occurences per word and sum them
total_number_all_words = original_tweets.value_counts().sum()
print("Total number of all words (including repetitions):", total_number_all_words)

# Find the number of all distinct words
total_number_all_unique_words = original_tweets.nunique()
print("Total number of all distinct words:", total_number_all_unique_words)

# Find the 10 most frequent words in the corpus
top_10_words = original_tweets.value_counts().nlargest(10)
print("10 most frequent words in the corpus:", top_10_words)

# Function to remove short words
def remove_shortwords(x):
    if len(x) > 2:
        return x

# Remove_the_short_words
original_tweets_without_short_words =  original_tweets.apply(lambda x: remove_shortwords(x))

# List of stopwords
stopwords = requests.get("https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt").content.decode("utf-8").split("\n")

# Create a mask to apply to the series
mask = np.logical_not(original_tweets_without_short_words.isin(stopwords))

# Apply the mask to the series in order to remove the stopwords
original_tweets_cleaned = original_tweets_without_short_words[mask]

# Count the number of occurences per word and sum them in the modified corpus
total_number_all_words_modified_corpus = original_tweets_cleaned.value_counts().sum()
print("Total number of all words (including repetitions) in the modified corpus:", total_number_all_words_modified_corpus)

# Find 10 most frequent words in the modified corpus
top_10_words_without_short_words = original_tweets_cleaned.value_counts().nlargest(10)
print("10 most frequent words in the modified corpus:", top_10_words_without_short_words)

# Take the original tweets cleaned/without short words and stop words
# Count them and sort them by ascending order
line_chart_data = original_tweets_cleaned.value_counts(ascending = True)

# Plot the data
# The X axis being the words
# The Y axis being the fraction of documents in which a word appear
plt.plot(range(len(line_chart_data)), line_chart_data.values/len(df1))

# Rename the axes
plt.xlabel("Words")
plt.ylabel("Frequencies")

# Show the plot
plt.show()

# Read the Corona_NLP_train.csv
df5 = pd.read_csv(r"Corona_NLP_train.csv", sep = ",", engine="python", encoding = "latin-1")

# Store the corpus of the original tweets in a numpy array
X_numpy = df5["OriginalTweet"].to_numpy()

# Store the sentiment in a numpy array
Y_numpy = df5["Sentiment"].to_numpy()

# Initialize the countVectorizer
vectorizer = CountVectorizer()

# Produce a sparse representation of the term-document matrix with a CountVectorizer
X = vectorizer.fit_transform(X_numpy)

# Initialize the Multinomial Bayes Classifier
classifier = MultinomialNB()

# Fit the classifier with the data from the CountVectorizer and the sentiment data
classifier.fit(X, Y_numpy)

# Compute the error rate of the classifier
error_rate = 1 - classifier.score(X, Y_numpy)
print(error_rate)

end = time.time()

print("Time taken by code:", end - start)
