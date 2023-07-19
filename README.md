# Traffic

## Experimentation Process

### CV2
First, I tried to read the file using `cv2.imread` with different flags:
- 1 for `IMREAD_COLOR`
- 0 for `IMREAD_GRAYSCALE`
- -1 for `IMREAD_UNCHANGED`

Using the flag -1, the files were read in approximately 4.3 seconds.*
Using the flag 0, the files were read in approximately 3.8 seconds.*
Using the flag 1, the files were read in approximately 4.5 seconds.*

"*" - values calculated on my computer (Windows 11, i7-8565U)

### TensorFlow

First, I used a convolutional neural network (from the CS50AI Notes), and I obtained the following results:
333/333 - 1s - loss: 3.5051 - accuracy: 0.0509 - 1s/epoch - 4ms/step

By adding another hidden layer with 128 units, I managed to achieve very similar results:
333/333 - 1s - loss: 3.4929 - accuracy: 0.0551 - 1s/epoch - 4ms/step

By keeping two hidden layers but changing to 64 units, I finally achieved better results. So, I doubled the number of hidden layers (4 layers, 64 units each), and the results improved even further:
333/333 - 1s - loss: 0.4104 - accuracy: 0.8953 - 1s/epoch - 4ms/step

Following this logic, I decided to reduce the number of units to 32 each and observe the results. However, I obtained worse results (333/333 - 1s - loss: 2.0174 - accuracy: 0.3126 - 1s/epoch - 4ms/step). Therefore, I doubled the number of hidden layers to 4 layers, with 32 units each, and I will compare the results later with 64 units each.

Results for 8 layers with 32 units each:
333/333 - 1s - loss: 0.8660 - accuracy: 0.7047 - 1s/epoch - 4ms/step
Results for 8 layers with 64 units each:
333/333 - 1s - loss: 0.4633 - accuracy: 0.8758 - 1s/epoch - 4ms/step

After reaching this point, I realized that further improvements were not significant. I decided to make one last change and set the number of units to the number of categories.

Furthermore, I added one more convolutional layer, and it yielded the best results so far:
333/333 - 2s - loss: 0.3315 - accuracy: 0.9077 - 2s/epoch - 5ms/step

To test the effect, I decided to revert to the original hidden layers:
- 1 layer with 128 units: 333/333 - 2s - loss: 0.2543 - accuracy: 0.9324 - 2s/epoch - 5ms/step
- 2 layers with 128 units: 333/333 - 2s - loss: 0.1829 - accuracy: 0.9586 - 2s/epoch - 5ms/step
- 2 layers with 64 units: 333/333 - 1s - loss: 0.2064 - accuracy: 0.9493 - 1s/epoch - 4ms/step
- 4 layers with 128 units: 333/333 - 2s - loss: 0.1692 - accuracy: 0.9552 - 2s/epoch - 5ms/step
- 4 layers with 64 units: 333/333 - 1s - loss: 0.2346 - accuracy: 0.9337 - 1s/epoch - 4ms/step
- 8 layers with 128 units: 333/333 - 2s - loss: 0.2533 - accuracy: 0.9361 - 2s/epoch - 5ms/step

So far, using two convolutional layers and four hidden layers with 128 units each, I obtained the best results. Adding another convolutional layer did not improve the results significantly.

Finally, I experimented with different `cv2.imread` flags. The best result was achieved when converting the image to grayscale:
333/333 - 2s - loss: 0.1414 - accuracy: 0.9682 - 2s/epoch - 5ms/step

Conclusion: The results improve when input pictures are read in grayscale (in this case, color is irrelevant). Adding an extra convolutional layer and a pooling layer increases accuracy and decreases loss. Increasing the number of hidden layers also improves accuracy and decreases loss (up to 4 layers, after which there is no noticeable difference). There was no significant difference noticed when changing the dropout value.

## Specification

### The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category number for each of the corresponding images in the images list.
Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
### The get_model function should return a compiled neural network model.
You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
- different numbers of convolutional and pooling layers
- different numbers and sizes of filters for convolutional layers
- different pool sizes for pooling layers
- different numbers and sizes of hidden layers
- dropout
