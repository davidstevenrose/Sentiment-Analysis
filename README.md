# Sentiment-Analysis
## Introduction
This repository explores NLP sentiment analysis by vectorizing data and training the machine. Topics present in the code include word tokenizing, parts-of-speech tagging, uni-grams, and bag-of-words (vectorizing). 

## Usage
This repository comes with a pickle file that already has a trained classifier. This classifier utilizes the provided dataset, but the dataset itself is not included in this repo. It can be found at\[1\]. The main script in this repo will train the classifier with the entire set, which I have done the liberty of and provided the pickle file for the classifier. <br/>
Once training completes, the classifier will test itself with a testing set, and disaply its accuracy. You may call def 'retrain_clsfyr' to retrain the classifier for better accuracy (this will also create a new pickle file and a backup of the old file). <br/>
The script will then ask for you to input a review. After input, the script will respond to whether it thinks your input is a good review or a bad one.<br/>
<b>Note:</b> I noticed that the machine tends to make better decisions with larger user inputs than succinct ones. For example, typing 'it was good' will most likely lead the macnine to decide 'bad review'.

## Citations
\[1\]Bo Pang and Lillian Lee, "Movie Review Data", June, 2004, v2.0
URL: http://www.cs.cornell.edu/people/pabo/movie-review-data
