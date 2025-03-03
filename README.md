# Iris.datapull

import pickle, joblib, time, os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import tensorflow as tf
import torch

# Load Dataset & Train Model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Measure Pickle Serialization
start = time.time()
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
pickle_time = time.time() - start
pickle_size = os.path.getsize("model.pkl")

# Measure Joblib Serialization
start = time.time()
joblib.dump(model, "model.joblib")
joblib_time = time.time() - start
joblib_size = os.path.getsize("model.joblib")

# Measure ONNX Serialization
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    start = time.time()
    f.write(onnx_model.SerializeToString())
onnx_time = time.time() - start
onnx_size = os.path.getsize("model.onnx")

# Measure TensorFlow Serialization
tf_model = tf.keras.Sequential([tf.keras.layers.Dense(3, activation='softmax')])
tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
start = time.time()
tf_model.save("tf_model")
tf_time = time.time() - start
tf_size = sum(os.path.getsize(os.path.join("tf_model", f)) for f in os.listdir("tf_model"))

# Measure TorchScript Serialization (PyTorch)
dummy_input = torch.randn(1, 4)
scripted_model = torch.jit.trace(torch.nn.Linear(4, 3), dummy_input)
start = time.time()
torch.jit.save(scripted_model, "model.pt")
torch_time = time.time() - start
torch_size = os.path.getsize("model.pt")

# Output Results
results = {
    "Method": ["Pickle", "Joblib", "ONNX", "TensorFlow", "TorchScript"],
    "Save Time (s)": [pickle_time, joblib_time, onnx_time, tf_time, torch_time],
    "File Size (KB)": [pickle_size / 1024, joblib_size / 1024, onnx_size / 1024, tf_size / 1024, torch_size / 1024]
}
import pandas as pd
df = pd.DataFrame(results)
print(df)
