import tensorflow as tf

# --- 1. Załaduj MNIST ---
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- 2. Zbuduj model ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. Trenuj model ---
model.fit(x_train, y_train, epochs=5)

# --- 4. Ewaluacja ---
model.evaluate(x_test, y_test)

# --- 5. Zapis modelu ---
model.save("mnist_model.keras")