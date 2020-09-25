from flask import Flask, request, render_template

from train import predict_label

app = Flask(__name__)
app.debug = True


@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    text = request.form['text']
    response = predict_label(text)
    return render_template('index.html', response=response.upper())


if __name__ == '__main__':
    app.run()