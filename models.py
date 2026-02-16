import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# Below contains the code for the ResNet Model, adapted for 1D time-series data.
def residual_block(x, filters, kernel_size=3):
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    return x


def build_resnet_1d(input_length, num_classes, learning_rate=1e-3):
    inputs = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)

    x = residual_block(x, 32)
    x = layers.MaxPooling1D(2)(x)

    x = residual_block(x, 64)
    x = layers.MaxPooling1D(2)(x)

    x = residual_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model


#Below contains the code for the EfficientNet Model, 
def build_effnet(img_size, num_classes, learning_rate=1e-4):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3),
    )

    base_model.trainable = True

    # Freeze lower layers, fine-tune top layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(base_model.input, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model
