from flask import Flask
from flask_restful import reqparse, Resource, Api
import pickle
import dill
import ktrain

app = Flask(__name__)
api = Api(app)

post_args = reqparse.RequestParser()
post_args.add_argument("text", type=str, help="text is required", required=True)

trans = dill.load(open('Models/NewsClassifier/Transformer.pickle', 'rb'))
vect = pickle.load(open('Models/NewsClassifier/Vectorizer.pickle', 'rb'))
selec = pickle.load(open('Models/NewsClassifier/Selector.pickle', 'rb'))
text_classifier = pickle.load(open('Models/NewsClassifier/Classifier.pickle', 'rb'))

polarity_classifier = ktrain.load_predictor('Models/BertPolarityClassifier')


class NewsPrediction(Resource):

    @staticmethod
    def post():
        text = post_args.parse_args()['text']
        text_trans = trans(text)
        text_vec = vect.transform(text_trans)
        text_sel = selec.transform(text_vec)
        text_class = int(text_classifier.predict(text_sel))
        return text_class


class Polarity(Resource):

    @staticmethod
    def post():
        text = post_args.parse_args()['text']
        polarity = polarity_classifier.predict(text)
        if polarity == "Positive":
            return 1
        if polarity == "Negative":
            return -1
        if polarity == "Neutral":
            return 0


api.add_resource(NewsPrediction, "/NewsOrNot")
api.add_resource(Polarity, "/Polarity")

if __name__ == '__main__':
    app.run()
