import numpy as np
import tensorflow as tf
import valohai


def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])


default_inputs = {'dataset': 'datum://01811510-5839-5b7f-6730-6ab8fc2dee12'}

input_path = valohai.inputs('dataset').path()

with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
model.fit(x_train, y_train, epochs=valohai.parameters('epochs').value, callbacks=[callback])

test_accuracy, test_loss = model.evaluate(x_test, y_test, verbose=2)

with valohai.logger() as logger:
    logger.log('test_accuracy', test_accuracy)
    logger.log('test_loss', test_loss)

output_path = valohai.outputs().path('model.h5')
model.save(output_path)
