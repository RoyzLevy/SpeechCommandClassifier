# SpeechCommandClassifier
ML project for speech command classifications

The model’s architecture is a deep neural network with 3 layers of convolution. After each layer, Max Pooling layer is added in order to reduce the output size and after the 3 layers we added dropout as a regularization to prevent overfitting of the model to train set.
Convolution is chosen, rather than FC, in order to refer to local parts of the audio file (in order to capture more context than the word spoken in the file).
- Optimization: SGD optimizer.
- Batch size: 100.
- Activation function: ReLU (for non-linear learning).
- The output sizes of the convolution layers were selected to maximize the accuracy on the validation set, and they are:
Stratum 0 - 32. Stratum 1 - 64. Stratum 2 - 43.
- Number of epochs: 10 (after 10 epochs no significant improvement was made in the minimization of the loss, and we are still not reaching a state of overfitting).
- Learning Rate: constant of 0.05 .
- Size of the hidden layers: 1000 and 150.
- Final output size of the model: 30 (as the number of labels).

Findings (after 10 epochs):
- Avg. Loss: 0.0037
- Accuracy: 90%

# Usage Instructions
* Local usage - might need GPU for training / configure smaller number of epochs (results may vary accordingly).
1. git clone https://github.com/RoyzLevy/SpeechCommandClassifier.git (or SSH)
2. pip install -r requirements.txt
3. Get commands sound files from any available dataset (for example Kaggle's Synthetic Speech Commands Dataset: https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset)
4. Divide the data to train, validation and test directories (example for heirarchy provided)
5. run classifier_main.py and get the results of the trained model: "test_y" file of the predictions on the test set.
