# Deep Learning Repository

I created this repository to document my experiments and practical implementations as I transitioned from classical Machine Learning into Deep Learning. It covers neural network foundations, computer vision with Convolutional Neural Networks (CNNs), sequence modeling with Recurrent Neural Networks (RNNs), and a tabular classification mini-project.

## About

This repository represents the second stage of my AI engineering path (Machine Learning -> Deep Learning -> Generative AI). After understanding classical algorithms, I wanted to explore how artificial neural networks learn representations directly from data.

The notebooks here focus on implementing fundamental architectures using TensorFlow and Keras, understanding the role of activation functions and optimizers, diagnosing training dynamics through loss curves, and comparing deep architectures against classical baselines.

## Learning Path / Topics Covered

1. **Neural Network Fundamentals & Feedforward Networks**
   - Data normalization (Min-Max scaling vs. standardization in deep learning), building multi-layer perceptrons with `keras.Sequential` and `layers.Dense`, choosing activation functions (`relu`, `sigmoid`), binary cross-entropy loss, and SGD optimization with momentum.
   - Implementation: [`01_dl.ipynb`](./01_dl.ipynb)

2. **Computer Vision & Convolutional Neural Networks (CNNs)**
   - Image classification on the MNIST handwritten digit dataset (`28x28` grayscale).
   - Side-by-side architectural comparison:
     - Single-layer Perceptron baseline
     - Multi-layer Artificial Neural Network (ANN)
     - Convolutional Neural Network (`Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Softmax`)
   - Visualizing training vs. validation accuracy curves, confusion matrix generation, and sample prediction inspections.
   - Implementation: [`02_CNN.ipynb`](./02_CNN.ipynb)

3. **Sequence Modeling & Recurrent Neural Networks (RNNs)**
   - Text preprocessing for sequence modeling: text tokenization (`Tokenizer`), vocabulary indexing, padding sequences, dense representation using Keras `Embedding` layers, and configuring `SimpleRNN` layers (`return_sequences`, `return_state`).
   - Implementation: [`03_RNN_implement.ipynb`](./03_RNN_implement.ipynb)

## Repository Structure

```text
deep-learning/
├── 01_dl.ipynb
├── 02_CNN.ipynb
├── 03_RNN_implement.ipynb
├── mini-project/
│   ├── Iris.csv
│   └── Iris_prediction.ipynb
├── notes/
│   └── README.md
├── requirements.txt
└── README.md
```


## Mini Project: Iris Flower Classification with Neural Networks

- **Directory**: [`mini-project/`](./mini-project/)
- **Problem**: Multi-class classification of Iris species (Setosa, Versicolor, Virginica) using clinical petal and sepal measurements.
- **Workflow**:
  1. Exploratory data analysis, pairplots, and feature scaling with `StandardScaler` on `Iris.csv`.
  2. Encoding target classes using `LabelEncoder` and `to_categorical`.
  3. Establishing a baseline using Scikit-Learn's `Perceptron`.
  4. Constructing a Multi-Layer Perceptron (ANN) using Keras (`Dense` layers with ReLU and Softmax activations) in [`Iris_prediction.ipynb`](./mini-project/Iris_prediction.ipynb).
  5. Training with the Adam optimizer and categorical cross-entropy loss, followed by training and validation accuracy curve analysis.
- **Status**: Completed notebook experiment.

## My Handwritten Notes

I maintain handwritten notes covering mathematical backpropagation derivations, matrix operations for convolution and pooling, and sequence model state updates.

The structure for my handwritten notes is documented in the notes directory:
- [View Handwritten Notes Documentation](./notes/README.md)

Scanned pages are currently being organized and digitized topic by topic.

## Tools & Technologies

- **Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Data Manipulation & Preprocessing**: NumPy, Pandas, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook / IPython Kernel

## Key Learnings

- **Spatial feature hierarchies in CNNs**: While flat ANNs ignore spatial relationships between neighboring pixels, convolutional layers preserve 2D structure and extract localized edge, texture, and shape patterns with shared weights.
- **Activation and loss alignment**: Multi-class classification requires Softmax with categorical cross-entropy, whereas binary classification pairs with Sigmoid and binary cross-entropy.
- **Sequence representation**: Raw text must be converted from discrete tokens into continuous vector spaces via Embedding layers before sequential models can process temporal context.
- **Monitoring learning curves**: Plotting training and validation loss per epoch is essential to spot variance issues (overfitting) early.

## Current Status

The foundational deep learning implementations across dense networks, CNNs, RNNs, and the Iris mini-project are complete.

## What's Next

Following this repository, I moved forward in my learning path to **Generative AI** (Large Language Models, Embeddings, Prompt Engineering, LangChain LCEL, and Retrieval-Augmented Generation / RAG), located in the sibling `genai-learning` folder.
