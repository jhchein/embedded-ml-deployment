import os

import numpy as np
import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt
import math

from azureml.core import Run

run = Run.get_context()

np.random.seed(1)
tf.random.set_seed(1)

SAMPLES = 1000

MODELS_DIR = "outputs/"

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_TF = os.path.join(MODELS_DIR, "model.pb")
MODEL_NO_QUANT_TFLITE = os.path.join(MODELS_DIR, "model_no_quant.tflite")
MODEL_TFLITE = os.path.join(MODELS_DIR, "model.tflite")
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, "model.cc")

x_values = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES).astype(np.float32)
np.random.shuffle(x_values)
y_values = np.sin(x_values).astype(np.float32)

plt.plot(x_values, y_values, "b.")
run.log_image("Sine", plot=plt)
plt.clf()

y_values += 0.1 * np.random.randn(*y_values.shape)
plt.plot(x_values, y_values, "b.")
run.log_image("Sine with Jitter", plot=plt)
plt.clf()

TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

assert (x_train.size + x_validate.size + x_test.size) == SAMPLES

plt.plot(x_train, y_train, "b.", label="Train")
plt.plot(x_test, y_test, "r.", label="Test")
plt.plot(x_validate, y_validate, "y.", label="Validate")
plt.legend()
run.log_image("Train, Test and Validation Data", plot=plt)
plt.clf()

model = tf.keras.Sequential()
model.add(keras.layers.Dense(16, activation="relu", input_shape=(1,)))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=64,
    validation_data=(x_validate, y_validate),
)

SKIP = 100
loss = history.history["loss"][SKIP:]
epochs = range(SKIP + 1, len(loss) + SKIP + 1)
val_loss = history.history["val_loss"][SKIP:]

plt.plot(epochs, loss, "g.", label="Training loss")
plt.plot(epochs, val_loss, "b.", label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
run.log_image("Loss", plot=plt)
plt.clf()

mae = history.history["mae"][SKIP:]
val_mae = history.history["val_mae"][SKIP:]
plt.plot(epochs, mae, "g.", label="Training MAE")
plt.plot(epochs, val_mae, "b.", label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
run.log_image("MAE", plot=plt)
plt.clf()

loss = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
plt.title("Comparison of predictions and actual values")
plt.plot(x_test, y_test, "b.", label="Actual")
plt.plot(x_test, predictions, "r.", label="Predicted")
plt.legend()
run.log_image("Predictions vs Actuals", plot=plt)
plt.clf()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_no_quant_tflite = converter.convert()

with open(MODEL_NO_QUANT_TFLITE, "wb") as fh:
    fh.write(model_no_quant_tflite)


def representative_dataset():
    for i in range(500):
        yield ([x_train[i].reshape(1, 1)])


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

with open(MODEL_TFLITE, "wb") as fh:
    fh.write(model_tflite)

model_no_quant_size = os.path.getsize(MODEL_NO_QUANT_TFLITE)
print("Model is %d bytes" % model_no_quant_size)
model_size = os.path.getsize(MODEL_TFLITE)
print("Quantized model is %d bytes" % model_size)
difference = model_no_quant_size - model_size
print("Difference is %d bytes" % difference)

model_no_quant = tf.lite.Interpreter(MODEL_NO_QUANT_TFLITE)
model_quant = tf.lite.Interpreter(MODEL_TFLITE)

model_no_quant.allocate_tensors()
model_quant.allocate_tensors()

model_no_quant_input = model_no_quant.tensor(
    model_no_quant.get_input_details()[0]["index"]
)
model_no_quant_output = model_no_quant.tensor(
    model_no_quant.get_output_details()[0]["index"]
)
model_quant_input = model_quant.tensor(model_quant.get_input_details()[0]["index"])
model_quant_output = model_quant.tensor(model_quant.get_output_details()[0]["index"])

model_no_quant_predictions = np.empty(x_test.size)
model_quant_predictions = np.empty(x_test.size)

for i in range(x_test.size):
    model_no_quant_input().fill(x_test[i])
    model_no_quant.invoke()
    model_no_quant_predictions[i] = model_no_quant_output()[0]

    model_quant_input().fill(x_test[i])
    model_quant.invoke()
    model_quant_predictions[i] = model_quant_output()[0]

plt.title("Comparison of various models against actual values")
plt.plot(x_test, y_test, "bo", label="Actual values")
plt.plot(x_test, predictions, "ro", label="Original predictions")
plt.plot(x_test, model_no_quant_predictions, "bx", label="Lite predictions")
plt.plot(x_test, model_quant_predictions, "gx", label="Lite quantized predictions")
plt.legend()
run.log_image("Comparison of various models against actual values", plot=plt)

os.system(f"xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}")
