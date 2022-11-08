from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
import preprocessor as p
import joblib
import numpy
from sentence_transformers import SentenceTransformer

# Start of datacleaning function #
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
insensitive_num = re.compile(re.escape('number'), re.IGNORECASE)
insensitive_url = re.compile(re.escape('url'), re.IGNORECASE)
insensitive_email = re.compile(re.escape('e mail'), re.IGNORECASE)

def cleanhtml(raw_html):
    cleaned_html = []
    for html in raw_html:
        html = str(html)
        html = CLEANR.sub('', html)
        html = insensitive_num.sub('', html)
        html = insensitive_url.sub('', html)
        html = insensitive_email.sub('email', html)
        cleaned_html.append(html)
    return cleaned_html

contractions = pd.read_json("contractions.json", typ='series')
contractions = contractions.to_dict()
c_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_ \t]')
lm = WordNetLemmatizer()
def clean_texts(reviews):
    cleaned_texts = []
    for review in reviews:
        review = str(review)
        review = review.lower()
        review = BAD_SYMBOLS_RE.sub(' ', review)  # remove punctuation
        review = p.clean(review)  # using tweet-preprocessor to clean the data once again
        review = expandContractions(review)  # expand contraction
        stop_words = set(stopwords.words('english'))  # set the stop words
        word_tokens = nltk.word_tokenize(review)
        filtered_sentence = [lm.lemmatize(w) for w in word_tokens if not w in stop_words]
        review = ' '.join(filtered_sentence)
        remove_single_w = [w for w in review.split() if len(w) > 1]
        review = ' '.join(remove_single_w)
        cleaned_texts.append(review)
    return cleaned_texts
# End of datacleaning function #

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_nn = joblib.load(open("nn.pkl", "rb"))

    if request.method == 'POST':
        message = request.form['message']
        message = [message]
        dataset = {'message': message}
        data = pd.DataFrame(dataset)

        data["message"] = cleanhtml(data["message"])
        data["message"] = clean_texts([review for review in data["message"]])

        model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        X_emb = model.encode(data["message"])
        my_prediction = model_nn.predict(X_emb)
        my_prediction = my_prediction[0]
        message = [word.replace("\r","").replace("\n","").replace("['","").replace("']","") for word in message]
        message = message[0]
    return render_template('result.html', prediction=my_prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#    app.run()
