import os
from flask import Flask, request, redirect, url_for, render_template, flash
from flask_assets import Environment, Bundle
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

classes = ["dog", "cat"]
num_classes = len(classes)
IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

assets = Environment(app)
assets.url = app.static_url_path
scss = Bundle('stylesheet.scss', filters='pyscss', output='all.css')
assets.register('scss_all', scss)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

graph = tf.compat.v1.get_default_graph()



@app.route("/")
def home():
    return render_template("home.html")

@app.route('/index', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        model = load_model('./test.h5', compile=False)  # 学習済みモデルをロードする

        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                # 受け取った画像を読み込み、np形式に変換
                img = image.load_img(filepath, target_size=TARGET_SIZE)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                # images = np.vstack([[x]])
                # 変換したデータをモデルに渡して予測する
                result = model.predict(x)
                print(result[0])
                if result[0] > 0.5:
                    answer = "犬"
                else:
                    answer = "猫"
                # predicted = np.argmax(result[0])
                # pred_answer = "これは " + classes[predicted] + " です"
                pred_answer = "これはもしや…" + answer + "では？"

                return render_template("index.html", answer=pred_answer)

        return render_template("index.html", answer="")


if __name__ == "__main__":
    
    app.run(debug=True)