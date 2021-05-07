# -*- coding: utf-8 -*-
"""
Natural Language Processing using Python

Building a Spam Detection filter

@author: suvosmac
"""

# required libraries
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# we will get a list of all the lines of text messages using list comprehension

messages = [line.rstrip() for line in 
            open('/Volumes/Data/CodeMagic/Data Files/Udemy/smsspamcollection/SMSSpamCollection')]

# Check the length of the messages
print(len(messages))

# Check the first message
print(messages[0])

# A collection of texts is also sometimes called "corpus". 
# Let's print the first ten messages and number them using enumerate

for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print("\n")
    
'''
Due to the spacing we can tell that this is a TSV ("tab separated values") file, 
where the first column is a label saying whether the given message is a normal message 
(commonly known as "ham") or "spam". The second column is the message itself. 
(Note our numbers aren't part of the file, they are just from the enumerate call).
Using these labeled ham and spam examples, we'll train a machine learning model 
to learn to discriminate between ham/spam automatically. Then, with a trained model, 
we'll be able to classify arbitrary unlabeled messages as ham or spam.
'''

messages = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/Udemy/smsspamcollection/SMSSpamCollection',
                       sep = '\t', names = ['label','message'])

# Check the first few rows of the data frame
messages.head()

'''
Exploratory Data Analysis
A large part of NLP is feature engineering, ability to extract more features from the data
Better is the domain knowledge, more is ability to extract features
'''
messages.describe()
messages['label'].value_counts()

# Lets use groupby to run describe on labels, to get an idea what separates a spam vs ham
messages.groupby('label').describe()

# We will add one more column in the data, how long the text messages are
messages['length'] = messages['message'].apply(len)

# Now check the initial rows
messages.head()

# Plot the frequency distribution of message length
messages['length'].plot.hist(bins=150)
sns.distplot(messages['length'])

# We see some messages are really long, see some summary statistics
messages['length'].describe()

# If we want to see messages which are greater than 900
messages[messages['length'] > 900]['message']
# to see the entire message
messages[messages['length'] > 900]['message'].iloc[0]

# Lets see this length histogram by label through a facet grid
messages.hist(column='length',by='label',bins=60)
# Looking at the above we see spam messages in general have more characters

'''
How to remove punctuation and stopwords
'''
mess = 'Sample Message! Notice: it has no punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]
# The above returns a list of characters.. we will join them back
nopunc = ''.join(nopunc)
print(nopunc) # added blank space everywhere there was punctuation

from nltk.corpus import stopwords
stopwords.words('english')

# Split nopunc again as a list of all words
nopunc.split()

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess

# Now we will create a function which can be applied to the entire dataframe
def text_process(mess):
    """
    1. remove punc
    2. remove stopwords
    3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    return clean_words

# We will apply this in the first 5 text messages
messages['message'].head(5).apply(text_process)

'''
There are a lot of ways to continue normalizing this text. Such as Stemming or distinguishing by part of speech.
NLTK has lots of built-in tools and great documentation on a lot of these methods. 
Sometimes they don't work well for text-messages due to the way a lot of people tend to use 
abbreviations or shorthand, For example:
    
'Nah dawg, IDK! Wut time u headin to da club?'
versus
'No dog, I don't know! What time are you heading to the club?'

Some text normalization methods will have trouble with this type of shorthand and so I'll leave 
you to explore those more advanced methods through the NLTK book online.
For now we will just focus on using what we have to convert our list of words to an actual vector 
that SciKit-Learn can use.
'''

'''
Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to 
convert each of those messages into a vector the SciKit Learn's algorithm models can work with.
Now we'll convert each message, represented as a list of tokens (lemmas) above, into a 
vector that machine learning models can understand.

We'll do that in three steps using the bag-of-words model:
Count how many times does a word occur in each message (Known as term frequency)
Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

Each vector will have as many dimensions as there are unique words in the SMS corpus. 
We will first use SciKit Learn's CountVectorizer. This model will convert a collection of text documents 
to a matrix of token counts.
We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary 
(1 row per word) and the other dimension are the actual documents, in this case a column per text message.

                	Message 1	Message 2	...	Message N
Word 1 Count	       0        1               0
Word 2 Count	       0	        0	      ...	0
              ...  1     	2	      ...	0
Word N Count	       0      	1	      ...	1


Since there are so many messages, we can expect a lot of zero counts for the presence of that word in that document. 
Because of this, SciKit Learn will output a Sparse Matrix.
'''
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# print the total number of vocab words
print(len(bow_transformer.vocabulary_))

# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer
message4 = messages['message'][3]
print(message4)

# To see its vector representation
bow_message4 = bow_transformer.transform([message4])
print(bow_message4)
print(bow_message4.shape)

# It shows there are 7 unique words and two of them appear twice, to find out which one appear twice
print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])

'''
Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the 
entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts for the entire 
SMS corpus is a large, sparse matrix
'''
messages_bow = bow_transformer.transform(messages['message'])
print("Shape of the Sparse Matrix :",messages_bow.shape)
print("Amount of Non-zero occurences :",messages_bow.nnz)

# Now we will find sparsity, which is percentage of non-zero elements
sparsity = (100.0 * messages_bow.nnz/(messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))

"""
So what is TF-IDF?
TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often 
used in information retrieval and text mining. This weight is a statistical measure used to evaluate how 
important a word is to a document in a collection or corpus. The importance increases proportionally to the 
number of times a word appears in the document but is offset by the frequency of the word in the corpus. 
Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and 
ranking a document's relevance given a user query.
One of the simplest ranking functions is computed by summing the tf-idf for each query term; 
many more sophisticated ranking functions are variants of this simple model.
Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), 
aka. the number of times a word appears in a document, divided by the total number of words in that document; 
the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the 
documents in the corpus divided by the number of documents where the specific term appears.

TF: Term Frequency, which measures how frequently a term occurs in a document. 
Since every document is different in length, it is possible that a term would appear much more times in 
long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. 
the total number of terms in the document) as a way of normalization:
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all 
terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", 
may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while 
scale up the rare ones, by computing the following:
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

See below for a simple example.
Example:
Consider a document containing 100 words wherein the word cat appears 3 times.
The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents 
and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is 
calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product 
of these quantities: 0.03 * 4 = 0.12.

"""

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Now we will find the TF-IDF score on the fourth message we transformed earlier
tfidf4 = tfidf_transformer.transform(bow_message4)
print(tfidf4)

#To transform the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

# We'll go ahead and check what is the IDF (inverse document frequency) of the word "u" and of word "university"?
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

'''
Training the model
Now we can actually use almost any sort of classification algorithms. For a variety of reasons, 
the Naive Bayes classifier algorithm is a good choice.
'''

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])

# Let's try classifying a single random message and checking how we do
print('predicted: ',spam_detect_model.predict(tfidf4)[0])
print('expected:', messages['label'][3])

'''
Model Evaluation
Now we want to determine how well our model will do overall on the entire dataset. 
Let's begin by getting all the predictions:
'''
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))

'''
We will re-do this using train-test split
'''
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

'''
We will use SciKit Learn's pipeline capabilities to store a pipeline of workflow. 
This will allow us to set up all the transformations that we will do to the data for future use. 
'''
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Now we can directly pass message text data and the pipeline will do our pre-processing for us!
pipeline.fit(msg_train,label_train)
# run the predictions
predictions = pipeline.predict(msg_test)

# run classification report
print(classification_report(predictions,label_test))