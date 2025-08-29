from flask import Flask, render_template, request, redirect
from helper import preprocessing, vectorization, get_prediction

app = Flask(__name__)

data = dict()
reviews = []
positive = 0
negative = 0

@ app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html',data = data)

@ app.route("/",methods=['POST']) # This methad is run when the after press the submit
def my_post():
    text = request.form['text']
    preprocessed_text = preprocessing(text)
    vectorized_text = vectorization(preprocessed_text)
    prediction = get_prediction(vectorized_text)

    if prediction == 'Negative':
        global negative
        negative = negative + 1
    else:
        global positive
        positive = positive +1
    reviews.insert(0,text)
    return redirect(request.url)




if __name__ == "__main__":
    app.run()