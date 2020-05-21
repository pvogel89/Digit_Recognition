import tensorflow as tf
import os
# Import own functions
from prepare_data import prepare_data
from create_model import create_model
import test

###################
# Load the Training and Validation Data
###################

dataset = tf.keras.datasets.mnist
[x_train, y_train, x_test, y_test] = prepare_data(dataset)

###################
# Define the model of the neural network
###################

model = create_model()
# Display the model's architecture
model.summary()
# Define loss function and compile model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

###################
# Train & save
###################

# create checkpoint in case training stops or shall be ctd at some point
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
# Train the model with the new callback
model.fit(x_train,
          y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[cp_callback])  # Pass callback to training
# Save model
model.save('saved_model/my_model')


###################
# Validate data
###################
print('Validate data with validation test set')
loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

