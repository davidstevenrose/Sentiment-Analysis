import nltk
from nltk.classify.scikitlearn import SklearnClassifier
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string, random, os, pickle
from MyClassifier import MyClass
from os.path import exists

'''adjectives are a better way of understanding sentiment review, so lets make a bag of adjectives
j is adjec, r is adverb, v is verb'''
allowed_word_types = ["J"]
stop_words = list(set(stopwords.words('english')))

'''
retrains the classifier with the same NLP learning algorithm or a new algorithm
param sk an object of a NLP learning algorithm to use. Uses a naive Bayes classifier by default
param force forces the classifier to utilize the latest training, regardless if the accuracy on
the testing set is better or worse than before.
'''
def retrain_clsfyr(sk = None, force = False):
    if exists("ReviewClassifier.pickle.bak"):
        os.remove("ReviewClassifier.pickle.bak")
    os.rename("ReviewClassifier.pickle", "ReviewClassifier.pickle.bak")
    clsfyr.retrain()
    with open("ReviewClassifier.pickle", "wb") as f:
        pickle.dump(clsfyr, f)

def build_classifier(posFile, negFile, clas = None):
    files_pos = os.listdir(posFile)
    files_pos = [open(posFile+f, 'r',encoding='utf8').read() for f in files_pos]
    files_neg = os.listdir(negFile)
    files_neg = [open(negFile+f, 'r',encoding='utf8').read() for f in files_neg]
    
    '''uncomment the following two lines for better performance if you wish to change this script'''
    #files_pos = files_pos[0:250]
    #files_neg = files_neg[0:250]

    all_words = []
    documents = []    

    index = 1
    for p in files_pos:
        #create list of tuples where the first element of each tuple is a review and second is the label
        documents.append( (p, "pos"))
    
        #filter out punctuations
        filtered = p.translate(str.maketrans('', '', string.punctuation))
    
        #tokenize
        tokenized = word_tokenize(filtered)
    
        #filter out stopwords
        stopped_tokenized = [w for w in tokenized if not w in stop_words]
    
        #parts of speech tagging
        pos = nltk.pos_tag(stopped_tokenized)
    
        #out list of all adjectives identified by the allowed word types declared in preprocessing
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    
        print(str(index), "done")
        index+=1

    for p in files_neg:
        #create list of tuples where the first element of each tuple is a review and second is the label
        documents.append( (p, "neg"))
    
        #filter out punctuations
        filtered = p.translate(str.maketrans('', '', string.punctuation))
    
        #tokenize
        tokenized = word_tokenize(filtered)
    
        #filter out stopwords
        stopped_tokenized = [w for w in tokenized if not w in stop_words]
    
        #parts of speech tagging
        neg = nltk.pos_tag(stopped_tokenized)
    
        #out list of all adjectives identified by the allowed word types declared in preprocessing
        for w in neg:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    
        print(str(index), "done")
        index+=1
    
    '''perform a map reduce for frequency distribution'''
    all_words = nltk.FreqDist(all_words)

    #list X most frequent words
    word_features = list(all_words.keys())[:500]

    '''
    function to create a dictionary of features for each review in the list document
    let the keys be the words in word_features
    let the value of each word be true of false to represent pos or neg
    this is vectorizing each review for the machine
    '''
    def vectorize_features(document):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features

#create features for each review. each set format is tuple (vectorized list, pos/neg)
    feature_sets = [(vectorize_features(review), categ) for (review, categ) in documents]

    random.shuffle(feature_sets)

    training_set = feature_sets[:len(feature_sets)-100]
    testing_set = feature_sets[len(feature_sets)-100:]
    
    if clas:
        obj = SklearnClassifier(clas)
        classifier = obj.train(training_set)
        return MyClass(classifier, training_set, testing_set, word_features)
    else:
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return MyClass(classifier, training_set, testing_set, word_features)

clsfyr = None
try:
    clsfyr = pickle.load(open("ReviewClassifier.pickle", "rb"))
except (OSError, IOError) as e:
    clsfyr = build_classifier('C:\\Users\\drose.CORP\\eclipse-workspace\\NLP Project\\train\\pos\\',
                               'C:\\Users\\drose.CORP\\eclipse-workspace\\NLP Project\\train\\neg\\')
    pickle_out = open("ReviewClassifier.pickle", "wb")
    pickle.dump(clsfyr, pickle_out)


'''now lets try to predict if a review is positive or negative'''
'''uncomment the following line to retrain the classifier'''
#retrain_clsfyr()

print("Classifier accuracy percent:", clsfyr.getAccuracy()*100)
#clsfyr.getClassifier().show_most_informative_features(15)

review = input("Type the review that you would like judged.\n")
fs = [(clsfyr.vectorize(review), "pos")]
accur = nltk.classify.accuracy(clsfyr.getClassifier(), fs)
print("Good review" if accur==1.0 else "Bad Review")

'''there are several classifier classes, but the most accurate is MultinomialNB

summarily, the MultinomialNB class is the best class to use when performing sentimental analysis on reviews.'''
