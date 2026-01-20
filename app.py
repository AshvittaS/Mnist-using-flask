from flask import Flask , request , jsonify
from tensorflow.keras.models import load_model
import numpy as np
#load model
model=load_model("mnist.h5")
app=Flask(__name__)
@app.route("/")
def home():
    return app.send_static_file("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    data=request.json["image"]
    img=np.array(data).reshape(-1,28,28,1).astype("float32")/255.0
    predict=model.predict(img)
    digit=int(np.argmax(predict,axis=1)[0])
    return jsonify({"digit":digit})
if __name__ == "__main__":
    app.run(debug=True)
