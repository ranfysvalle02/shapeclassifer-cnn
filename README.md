# Shape Up: Classifying Geometric Patterns with CNNs

![](https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

__Image credit to [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)__

## What is a Convolutional Neural Network (CNN)?

A Convolutional Neural Network (CNN) is a powerful type of neural network designed to process grid-like data, such as images. CNNs are particularly well-suited for image recognition tasks because they can automatically detect hierarchical features (edges, textures, objects) within an image.

A convolutional layer applies a series of filters to the input. Each filter is responsible for learning to recognize a specific feature in the image, such as edges, corners, or color blobs. The output of the convolutional layer is a feature map that highlights the areas in the image where those features were detected.

After the convolutional layers, the data is flattened and passed through one or more fully connected layers, which perform classification based on the detected features.

A **fully connected (dense) layer** connects every neuron in one layer to every neuron in the next layer, creating a dense matrix of weights. The number of parameters is a function of the number of input and output neurons.

CNNs consist of multiple layers, including **convolutional layers**, which apply filters to the input. Afterward, the output from these layers—known as **feature maps**—is passed through **pooling layers** and finally **fully connected layers**, which use the detected features to make a prediction or classification.

## Understanding Layers in a CNN

Layers are the fundamental building blocks of any neural network, and their configuration determines the network's ability to learn and generalize from data. In a Convolutional Neural Network (CNN), different types of layers serve different purposes:

- **Convolutional Layers**: These layers apply filters (kernels) to the input data to detect specific features like edges, shapes, or textures. Kernels in Convolutional Neural Networks (CNNs) are small, learnable matrices that are applied to the input data (typically images) to extract features. Think of a kernel as a small, specialized detector. It's designed to identify a particular feature within the image, such as an edge, corner, or texture. The filters slide across the input image, performing element-wise multiplications and creating feature maps. Each layer's filters capture more complex features as we move deeper into the network.
  
- **Pooling Layers**: After the convolutional layers, pooling layers are used to reduce the spatial dimensions of the feature maps. This not only decreases computational requirements but also helps prevent overfitting by summarizing the features. Max pooling, the most common type, selects the largest value in a window of the feature map, preserving important information while discarding irrelevant details.

- **Fully Connected Layers**: After the convolution and pooling layers have extracted meaningful features, the data is flattened and passed through one or more fully connected (dense) layers. These layers perform the final classification or regression task by connecting every neuron to every other neuron from the previous layer.

In our model, we used:
- Two **convolutional layers** with ReLU activation and max pooling, which help learn the shapes (squares, triangles).
- A **dropout layer** to prevent overfitting by randomly "dropping out" neurons during training.
- A **fully connected layer** to classify the shapes based on the learned features.

Each layer contributes uniquely to the learning process. Convolutional layers extract the raw visual features, pooling layers downsample them, and fully connected layers use those features to classify or make predictions.

In Convolutional Neural Networks (CNNs), the transformations applied in the convolutional layers are essential for learning and extracting features from images. These transformations occur in a few stages:

### 1. Convolutions: Learning Features

The primary operation in a CNN is **convolution**. In our `SimpleShapeClassifier`, we used two convolutional layers (`conv1` and `conv2`), each of which applies a set of filters (also known as kernels) to the input image. These filters are responsible for detecting patterns like edges, corners, and textures in the data.

When a convolution operation is applied:
- A filter of a fixed size (e.g., 3x3) slides over the image and performs element-wise multiplication between the filter and the patch of the image it’s currently over. 
- The result of this multiplication is summed up and forms one pixel of the resulting feature map.
- Different filters extract different features: some may detect horizontal edges, while others may focus on vertical edges, shapes, or textures.
  
In our model, the first convolutional layer (`conv1`) uses 32 filters, and the second (`conv2`) uses 64 filters. These layers automatically learn to extract meaningful features from the training data as the model optimizes its parameters.

### 2. Activation Function: Introducing Non-linearity

![](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/05/relu_activation.png?lossy=2&strip=1&webp=1)

__Image credit to [Convolutional Neural Networks (CNNs) and Layer Types](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/)__

After the convolution operation, we apply an **activation function** to introduce non-linearity. In our case, we use the **ReLU (Rectified Linear Unit)** activation function, which ensures the network can learn complex patterns. ReLU works by zeroing out negative values in the feature maps while keeping positive values unchanged. This helps the model focus on significant features while ignoring less important ones.

For example:
- If a certain region of the image strongly matches a feature (such as an edge or corner), ReLU will keep that high value.
- If the region does not match, ReLU will zero out that region’s contribution, helping the model focus on areas with more meaningful information.

### 3. Pooling: Reducing Dimensionality

![](https://media.geeksforgeeks.org/wp-content/uploads/20190721025744/Screenshot-2019-07-21-at-2.57.13-AM.png)
__Image credit to [CNN | Introduction to Pooling Layer](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/)__


After each convolution and ReLU activation, we perform **max pooling**. Pooling layers downsample the feature maps, reducing the spatial size while retaining the most important information. In max pooling, we slide a window (e.g., 2x2) across the feature map and take the maximum value from each window.

This process:
- Reduces the computational complexity by shrinking the data.
- Makes the model more robust to small translations, distortions, or variations in the input image.
- Helps prevent overfitting by summarizing the most important features.

In our model, after each convolutional layer, we apply max pooling with a 2x2 window, halving the spatial dimensions of the feature maps. This transformation allows the CNN to focus on the most prominent features while discarding unnecessary detail.

### 4. Flattening: Preparing for Fully Connected Layers

After the final convolutional and pooling layers, we **flatten** the feature maps into a 1D vector before passing them to the fully connected (dense) layers. This vector contains all the learned features from the image, which the fully connected layers use to make the final classification.

In our `SimpleShapeClassifier`, the input to the fully connected layer (`fc1`) is a flattened version of the output from the second pooling layer. This transforms the 3D feature map (with dimensions 64 x 64 x 64) into a 1D vector of length 262,144 (64 * 64 * 64), which is then passed through the dense layer for classification.

## Why These Transformations Matter

These transformations enable the model to learn and extract hierarchical features from the input data. Initially, the convolutional layers capture low-level features (such as edges and simple shapes), and as the data progresses through deeper layers, the model learns more complex patterns (such as entire shapes or objects).

Without these transformations, the model wouldn’t be able to detect patterns effectively. Each step—from convolution to pooling to flattening—contributes to the CNN's ability to classify images based on the learned features. 

For instance, the model learns that a "square" is characterized by straight edges and right angles, while a "triangle" has three edges converging at specific angles. By detecting and refining these patterns through multiple layers, the CNN can predict which shape is in an image.

This combination of convolution, activation, and pooling helps the model generalize better, meaning it can make accurate predictions on new, unseen data, which is the ultimate goal of training a CNN.

## ReLU Activation and Max Pooling in CNNs

**ReLU Activation** and **Max Pooling** are two key components of Convolutional Neural Networks (CNNs) that play crucial roles in improving the network's performance.

### ReLU Activation

* **Purpose:** Introduces non-linearity into the network. Without non-linearity, a CNN would essentially be a linear function, unable to learn complex patterns.
* **Function:** Replaces negative values in the output of a layer with zero. Mathematically, ReLU(x) = max(0, x).
* **Benefits:**
    * Prevents vanishing gradient problem.
    * Faster computation than sigmoid or tanh functions.
    * Sparsity: Many neurons output zero, which can improve efficiency.

### Max Pooling

* **Purpose:** Reduces the spatial dimensions of the feature maps, making the network more computationally efficient and reducing overfitting.
* **Function:** Slides a window (e.g., 2x2) over the feature map and selects the maximum value within that window.
* **Benefits:**
    * Invariance to small translations and distortions.
    * Reduces the number of parameters, preventing overfitting.
    * Down-sampling without losing too much information.

**How ReLU and Max Pooling Work Together:**

1. **Convolutional Layer:** Applies filters to the input image, producing feature maps.
2. **ReLU Activation:** Applies ReLU to introduce non-linearity to the feature maps.
3. **Max Pooling:** Downsamples the feature maps, reducing their spatial dimensions.

**Example:**

Consider a 3x3 feature map:

```
1 2 3
4 5 6
7 8 9
```

Applying 2x2 max pooling with a stride of 2 would result in:

```
5 6
8 9
```

**Key Points:**

* ReLU and max pooling are often used together in CNNs.
* The choice of pooling size and stride can affect the network's performance.
* Other pooling techniques exist, such as average pooling, but max pooling is the most common.

By understanding ReLU and max pooling, you can better appreciate how CNNs extract features and make predictions on image data.

## Understanding Model Parameters

In machine learning, particularly in neural networks, parameters are the internal variables that the model learns through training. They are the essence of the model's learning capability - the more parameters a model has, the more complex patterns it can learn from the data.

In our `SimpleShapeClassifier` model, we have a large number of parameters - approximately 33.6 million. This might seem like a lot, but it's primarily due to the fully connected layer (`fc1`) in our network.

## Fully Connected Layer and Parameters

A fully connected layer, also known as a dense layer, is a layer in which each neuron is connected to every neuron in the previous layer. This results in a large number of parameters, specifically, the number of parameters in a fully connected layer is equal to the product of the number of input neurons and the number of output neurons.

In our case, the fully connected layer (`fc1`) is transforming a flattened version of the output from the previous layer (which has a size of 64 * 64 * 64 = 262,144 neurons) into a layer with 128 neurons. This results in 262,144 (input neurons) * 128 (output neurons) = approximately 33.6M parameters.

This might seem excessive, but this is common in CNNs, especially when transitioning from convolutional layers to dense layers, which require a large number of connections.

## What Does This Mean?

The large number of parameters gives our model a high learning capacity. This means it can learn complex patterns from the data, which is beneficial for tasks like image classification where the input data (images) can contain complex and varied patterns.

However, it's important to note that a high number of parameters can also lead to overfitting, where the model learns the training data too well and performs poorly on unseen data. To prevent this, we can use techniques like regularization and dropout.

In conclusion, the number of parameters in a model is a double-edged sword - while it allows the model to learn complex patterns, it can also lead to overfitting. Therefore, it's important to find a good balance and choose a model architecture that is complex enough to learn from the data, but not so complex that it overfits.

## Overfitting and How to Prevent It

**Overfitting** is a common problem in machine learning, especially when dealing with small datasets or complex models. It happens when a model learns not only the underlying patterns in the data but also the noise or minor variations, which do not generalize well to unseen data.

### Signs of Overfitting:
- The model performs exceptionally well on the training data but poorly on the validation or test data.
- The loss on the validation set increases after a certain point, even though the training loss continues to decrease.

### Solutions to Prevent Overfitting:
1. **Dropout**: This technique randomly disables neurons in the network during training, forcing the model to learn more robust features. In our CNN, we applied dropout after the second convolutional layer and before the fully connected layer to reduce reliance on any specific neurons.
   
2. **Data Augmentation**: You can create more data by applying random transformations to the images (e.g., rotations, flips, and translations). This makes the model more robust to variations in input data.

3. **Regularization (L2)**: Adding a penalty to the model's complexity by encouraging smaller weight values can help prevent overfitting. This is implemented by penalizing large weights during training.

4. **Early Stopping**: By monitoring the validation loss, you can stop training once the model starts overfitting. This prevents the model from learning noise in the data.

5. **Cross-validation**: Instead of using a single train-validation split, you can divide the dataset into multiple subsets and rotate through them, training and validating on different parts of the data. This helps ensure the model generalizes better across various subsets of the data.

## Why Simpler Models Can Be Better

For the shape classification task, a **simpler model** can often outperform a more complex one. This might seem counterintuitive, but here’s why:

- **Smaller datasets** or **simple tasks** (like classifying basic shapes) don’t require overly complex models. If the model is too complex, it may learn noise or irrelevant details from the data, leading to overfitting.
- In contrast, a simpler model focuses on learning only the most important patterns—such as the basic geometry of shapes—without overcomplicating the learning process.

By designing our CNN with only a few convolutional and dense layers, we ensure that the model is powerful enough to distinguish shapes without falling into the trap of overfitting. This simplicity also means **faster training times** and less computational resources, which are important considerations when scaling up a project.

In our shape classification task, the patterns to be learned (i.e., the shapes of the objects) are quite simple, so a complex model isn't necessary. A simpler model can learn these patterns effectively without the risk of overfitting.

## Breaking Down the Code

### Synthetic Data Generation

The first part of the code is dedicated to generating synthetic data. We define a function `generate_shape_image` that creates images of different shapes (circle, square, triangle) with random colors and sizes. This function is then used in `create_dataset` to generate a dataset of images and their corresponding labels. We train it only on `square` and `triangle`.

```python
SHAPES = ['square', 'triangle']
```

The advantage of synthetic data is that it allows us to generate as much data as we need, and we have full control over its characteristics. This is particularly useful when real-world data is scarce, expensive, or privacy-sensitive.

The data for the shape classification task is generated by **drawing shapes on a canvas**. In the code, this canvas is a blank image created using NumPy, which we fill with different shapes like squares, circles, and triangles.

### Here's how the data generation process works:

1. **Create a Blank Canvas:**
   The function `generate_shape_image` starts by creating an empty image (a black canvas) with a specific size, defined by `IMAGE_SIZE`. This canvas is essentially an array of zeros with dimensions `(256, 256, 3)`, where the three channels represent the RGB color space.

   ```python
   image = np.zeros(size + (3,), dtype=np.uint8)
   ```

   Each pixel in this canvas is initially black (`[0, 0, 0]`), representing the RGB values for a black color.

2. **Randomize Colors:**
   To make the dataset more interesting, we generate random colors for each shape. This is done by picking three random values between 0 and 255 (for RGB channels), which ensures that each shape has a different color when drawn on the canvas.

   ```python
   color = tuple(np.random.randint(0, 256, 3).tolist())
   ```

3. **Draw the Shapes:**
   Depending on the shape chosen (circle, square, or triangle), we use OpenCV functions to draw the shape onto the canvas:
   
   - **For a circle:** We randomly select a center point and radius and use `cv2.circle` to draw a filled circle.

     ```python
     cv2.circle(image, center, radius, color, thickness=-1)
     ```

   - **For a square:** We randomly choose the top-left corner and calculate the width and height. Then, we use `cv2.rectangle` to draw a filled rectangle (square in this case).

     ```python
     cv2.rectangle(image, tuple(top_left), tuple(bottom_right), color, thickness=-1)
     ```

   - **For a triangle:** We randomly select three points and use `cv2.fillPoly` to draw a filled triangle.

     ```python
     cv2.fillPoly(image, [points], color)
     ```

4. **Create the Dataset:**
   Once the shape is drawn, the image is added to the dataset, and a corresponding label is assigned (e.g., `0` for a square, `1` for a triangle).

   ```python
   dataset.append(image)
   labels.append(shapes.index(shape))
   ```

### Transforming Images for the Model
After generating the images, we apply transformations using `torchvision.transforms`. The images are converted into tensors, which are the format PyTorch uses to process data. This transformation normalizes pixel values and prepares the images for training the CNN.

```python
transform = transforms.Compose([ToTensor()])
dataset = torch.stack([transform(image) for image in dataset])
```

By creating a variety of shapes and applying random positions, sizes, and colors, this process generates a dataset that simulates real-world variability, helping the CNN learn to classify different shapes effectively.

### Data Preprocessing

The `split_dataset` function is used to split the dataset into a training set and a validation set. This is a common practice in machine learning to ensure that our model can generalize well to unseen data.

The `create_data_loaders` function creates PyTorch data loaders from the datasets. Data loaders are used to efficiently load and preprocess the data in batches during training.

### Model Definition

The `SimpleShapeClassifier` class defines our CNN model. It consists of two convolutional layers (`conv1` and `conv2`), a max pooling layer (`pool`), a dropout layer (`dropout`), and two fully connected layers (`fc1` and `fc2`). The `forward` method defines the forward pass of the network.

### Training and Testing

We use the PyTorch Lightning library to train and test our model. We create a `trainer` object and call its `fit` method to train the model on our training data. After training, we use the `test` method to evaluate the model on our test data.

### Prediction

Finally, we use our trained model to predict the shape of new unseen images. We generate a new image using our `generate_shape_image` function and pass it to the `predict_shape` function, which uses the model to predict the shape.

## Exploring New Possibilities with Generative AI and Transformer Model Architecture

Generative AI and Transformer models have opened up new possibilities in the field of machine learning. Generative AI models, like Generative Adversarial Networks (GANs), can generate new data that is similar to the input data. This can be used to generate more diverse synthetic data for training our models.

On the other hand, Transformer models, which were initially designed for natural language processing tasks, have shown great promise in other domains as well, including image classification. Transformer models, like Vision Transformers (ViT), treat an image as a sequence of patches and apply self-attention mechanisms to capture global dependencies between patches.

Incorporating these advanced models and techniques could further improve the performance of our shape classifier. For instance, we could use a GAN to generate more diverse shapes for training, or we could replace our CNN with a Vision Transformer to capture more complex patterns in the images.

## Troubleshooting Tips for CNNs

Building and training CNNs can come with various challenges. Here are some common problems and solutions:

- **Vanishing/Exploding Gradients**: This occurs when the gradients during backpropagation become too small or too large, causing slow convergence or instability. Use techniques like batch normalization or gradient clipping to stabilize training.

- **Imbalanced Data**: If one class in the dataset has more samples than another, the model may become biased towards predicting the majority class. You can address this by oversampling the minority class, undersampling the majority class, or using weighted loss functions.

- **Learning Rate Issues**: If the learning rate is too high, the model may not converge and might jump around the solution. If it's too low, the training process can be slow and get stuck in local minima. Use techniques like learning rate schedules or adaptive optimizers like Adam to fine-tune this parameter.

- **Model Not Learning**: If the training loss doesn't decrease, the issue could be incorrect model architecture, a learning rate that's too low, or a lack of sufficient data. Start with a simple model and ensure data is correctly preprocessed.

## Conclusion

In this blog post, we've explored the basics of Convolutional Neural Networks using a simple shape classification task as an example. We've seen how a CNN applies a series of filters to an image to detect features, and how these features are then used for classification. We've also discussed why simpler models can sometimes be more effective, especially when working with smaller datasets or simpler tasks.

Remember, machine learning is an iterative process that involves a lot of trial and error. Don't be discouraged if your model doesn't work perfectly the first time. Keep experimenting and tweaking until you get the results you want. Happy coding!
