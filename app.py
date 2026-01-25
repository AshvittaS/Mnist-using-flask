from flask import Flask , request , jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
#load model
model=load_model("mnist.h5")
app=Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return send_from_directory("static", "index.html")
@app.route("/predict",methods=['POST'])
def predict():
    data=request.json["image"]
    img=np.array(data).reshape(-1,28,28,1).astype("float32")/255.0
    predict=model.predict(img)
    digit=int(np.argmax(predict,axis=1)[0])
    return jsonify({"digit":digit})
if __name__ == "__main__":
    app.run(debug=True)
