**Neural networks In-Home Assignment 1**
University of Central Missouri
**Course: Neural Networks and Deep Learning**
Term: SUMMER 2025 Student Name: DANDABOINA SRI POOJITHA Student ID: [700765309]

Overview of the Assignment
This repository contains the solution for **Neural networks Assignment, covering fundamental deep learning concepts using TensorFlow. The assignment consists of four main tasks:

Tensor Operations & Reshaping
Loss Function Analysis & Hyperparameter Adjustment
Neural Network Training with TensorBoard Logging
Prerequisites
Before executing the script, ensure the following are installed:

Python (>=3.7)
TensorFlow (>=2.x)
NumPy
Matplotlib
To install dependencies, execute:

pip install tensorflow numpy matplotlib
Execution Instructions
Clone the repository and navigate to the directory:

git clone https://github.com/vamsidama/Neural networks.git
cd Neural networks
Run the script:

python Neural networks Assignment.jpynb
Launching TensorBoard
After training, logs can be viewed using TensorBoard:

tensorboard --logdir logs/fit/
#TASK 1 Task 1: Tensor Transformations & Reshaping

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
