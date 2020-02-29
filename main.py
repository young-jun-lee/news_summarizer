import flask
from flask import render_template, request
from wtforms import Form, validators, StringField
from sklearn.externals import joblib
import cloudpickle

main = flask.Flask(__name__)
main.config["DEBUG"] = True  #if the app is malformed you get an actual debug config, rather than just a bad gateway msg.

def get_txt(link):
    with open('txt_scraper.pkl', 'rb') as f:
        func = cloudpickle.load(f)
    txt = func(link)
    return txt

class LinkForm(Form):
    link = StringField('',[validators.DataRequired()])


@main.route('/')
def index():
    form = LinkForm(request.form)
    return render_template('home.html', form=form)


@main.route('/api', methods=["POST"])
def hello():
    form = LinkForm(request.form)
    if request.method == "POST" and form.validate():
        name = request.form['link']
        txt = get_txt(name)
        model = joblib.load('test_model')
        summary = model(txt)
        return render_template("about.html", name=summary)
    return render_template("home.html", form=form)

# @app.route('/', methods=['GET']) #routing the address to the /
# #the GET method specifies that you are sending information to the user
# #the POST method specifies that you are receiving information from the user
# def home():
#     return render_template("home.html")

@main.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


if __name__ == '__main__':
    main.run()
