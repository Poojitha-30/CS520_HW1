# Neural Networks – Home Assignment 1

## University of Central Missouri  
**Department:** Computer Science & Cybersecurity  
**Course:** CS5720 – Neural Networks and Deep Learning  
**Term:** Summer 2025  

---

## Student Information  
- **Name:** Sri Poojitha Dandaboina  
- **Student ID:** 700765309  
- **GitHub Repository:** [CS520_HW1](https://github.com/Poojitha-30/CS520_HW1)

---

## Overview
This assignment covers foundational concepts in deep learning using TensorFlow, including tensor manipulation, loss function analysis, and training a neural network with TensorBoard logging.

---

## Tasks Summary

###  Task 1: Tensor Manipulations & Reshaping
- Create a random tensor of shape (4, 6)
- Compute its rank and shape
- Reshape to (2, 3, 4) and transpose to (3, 2, 4)
- Perform broadcasting with a (1, 4) tensor and add to the larger tensor
- **Explanation included** on how broadcasting works in TensorFlow

###  Task 2: Loss Functions & Hyperparameter Tuning
- Define `y_true` and `y_pred` tensors
- Calculate **Mean Squared Error (MSE)** and **Categorical Cross-Entropy (CCE)**
- Modify predictions slightly to analyze loss variations
- Plot loss comparison using Matplotlib

###  Task 3: Neural Network Training with TensorBoard
- Load and preprocess the **MNIST dataset**
- Build and train a simple neural network for digit classification
- Enable **TensorBoard logging**
- Train the model for **5 epochs**
- Launch TensorBoard to monitor training and validation metrics

### Reflection Questions Answered:
- Observations on training vs. validation accuracy
- How TensorBoard helps detect overfitting
- Effects of increasing the number of epochs

---

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

## Install dependencies :

```bash
pip install tensorflow numpy matplotlib

 **How to Run**
Clone the repository and navigate to the directory:
git clone https://github.com/Poojitha-30/CS520_HW1.git
cd CS520_HW1

Open and run the notebook:
jupyter notebook CS520_HW1.ipynb

To launch TensorBoard:
tensorboard --logdir logs/fit/


Step 1: Generate a Random Tensor The code generates a 4×6 random tensor using TensorFlow: tensor_data = tf.random.uniform((4, 6)) This creates a tensor with values between 0 and 1.

Step 2: Find Tensor Rank and Shape The rank represents the number of dimensions (2D, 3D, etc.). The shape gives the size of each dimension. tensor_rank = tf.rank(tensor_data).numpy() tensor_shape = tensor_data.shape print(f"Rank: {tensor_rank}, Shape: {tensor_shape}") Expected Output: Rank: 2, Shape: (4, 6)

Step 3: Reshape and Transpose Reshaping converts the tensor from (4,6) → (2,3,4). reshaped_data = tf.reshape(tensor_data, (2, 3, 4)) Transposing swaps dimensions from (2,3,4) → (3,2,4). transposed_data = tf.transpose(reshaped_data, perm=[1, 0, 2]) Print before and after reshaping: print("Reshaped Tensor:", reshaped_data.numpy()) print("Transposed Tensor:", transposed_data.numpy())

Step 4: Broadcasting & Summation A small tensor (1,4) is created. TensorFlow broadcasts it to match the first tensor. small_data = tf.random.uniform((1, 4)) broadcasted_data = tf.broadcast_to(small_data, (4, 4)) result_data = tensor_data[:, :4] + broadcasted_data

Task 2: Compute and Compare Loss Functions

Step 1: Define True and Predicted Values y_actual = tf.constant([0.0, 1.0, 1.0, 0.0]) y_predicted = tf.constant([0.2, 0.9, 0.8, 0.1]) y_actual represents the ground truth. y_predicted represents model predictions.

Step 2: Calculate Loss Mean Squared Error (MSE): Measures average squared difference. mse_loss_fn = MeanSquaredError() mse_result = mse_loss_fn(y_actual, y_predicted).numpy() Categorical Cross-Entropy (CCE): Measures the difference between probability distributions. cce_loss_fn = CategoricalCrossentropy() cce_result = cce_loss_fn(tf.expand_dims(y_actual, axis=0), tf.expand_dims(y_predicted, axis=0)).numpy()

Step 3: Modify Predictions and Recalculate Loss Slightly adjust y_predicted to see loss variation. y_predicted_updated = tf.constant([0.1, 0.8, 0.9, 0.2]) mse_updated = mse_loss_fn(y_actual, y_predicted_updated).numpy() cce_updated = cce_loss_fn(tf.expand_dims(y_actual, axis=0), tf.expand_dims(y_predicted_updated, axis=0)).numpy()

Step 4: Plot Loss Function Values Visualize loss comparison using Matplotlib. plt.bar(["MSE", "CCE"], [mse_result, cce_result], color=['blue', 'red']) plt.xlabel("Loss Type") plt.ylabel("Loss Value") plt.title("MSE vs Cross-Entropy Loss Comparison") plt.show()

Task 3: Neural Network Training with TensorBoard

Step 1: Enable TensorBoard Logging log_directory = "logs/fit/" os.makedirs(log_directory, exist_ok=True) Creates a log directory for TensorBoard.

Step 2: Train Model with TensorBoard Callback model_tb = build_model() model_tb.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory, histogram_freq=1) model_tb.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), callbacks=[tb_callback]) Logs training accuracy and loss.

Step 3: Launch TensorBoard print("To launch TensorBoard, use: tensorboard --logdir logs/fit/")
