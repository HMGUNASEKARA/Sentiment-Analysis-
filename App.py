from flask import Flask, render_template, request, redirect
from helper import preprocessing, vectorization, get_prediction
from logger import logging

app = Flask(__name__)  # Here Define the app 
# In between Define and run we build the app 

logging.info("Flask sever started ")

data = dict()
reviews = []
positive = 0
negative = 0

@ app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info(" ============ opne home page ==============")

    return render_template('index.html',data = data)

@ app.route("/",methods=['POST']) # This methad is run when the after press the submit. Post kiyanne frond end eken submit karana eka
def my_post():
    text = request.form['text']
    logging.info(f'Text : {text}')

    preprocessed_text = preprocessing(text)
    logging.info(f'preprocessed_text : {preprocessed_text}')

    vectorized_text = vectorization(preprocessed_text)
    logging.info(f'vectorized_text : {vectorized_text}')

    prediction = get_prediction(vectorized_text)
    logging.info(f'prediction : {prediction}')


    if prediction == 'Negative':
        global negative
        negative = negative + 1
    else:
        global positive
        positive = positive +1
    reviews.insert(0,text)
    return redirect(request.url)




if __name__ == "__main__":  # In here we run the app 
    app.run()