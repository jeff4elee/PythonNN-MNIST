import random

# Third-party libraries
import numpy as np
import mnist_loader
from PIL import Image
from PIL import ImageOps
from scipy import ndimage
from skimage.morphology import medial_axis
from skimage.morphology import erosion, dilation

class Network(object):

    training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #print self.weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            #else:
                #print "Epoch {0} complete".format(j)
        return self

    def update_mini_batch(self, mini_batch, eta):
        #set gradients of biases and weights to same shaped matrices
        #as biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #for each training input/expected output in the
        #mini_batch, add the appropiate change of biases/weights to the
        #gradients
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def feedImage(self, image):
        '''takes an image and attempts to predict the number present using
            a neural network and pre-processing methods'''
        im = []
        image = self.gradual_normalization(image)
        (x, y) = ndimage.measurements.center_of_mass(np.array(image))
        print (x,y)
        large_window = Image.new("L", (28,28), "white")
        large_window.paste(image, (int(14.5-x),int(14.5-y)))
        image_matrix = self.standardize_image(large_window)/255
        print image_matrix
        
        image_edt = ndimage.distance_transform_edt(image_matrix)
        dist_values = [i for i in image_edt]
        thick_point = np.amax(dist_values)
        image_edt = np.float64(image_edt)/thick_point
        image_matrix = image_edt
        
        print image_matrix
        input_vector = np.reshape(image_matrix, (784, 1))
        result = self.feedforward(input_vector)
        max_index = 0
        for i in range(len(result)):
            if(result[max_index] < result[i]):
                max_index = i
        return max_index


    def gradual_normalization(self, image):
        '''adjusts image before resizing it to a 20x20 image, then normalizes the 20x20 image's stroke size'''        
        image = image.resize((250, 250), Image.ANTIALIAS)
        image = self.stroke_normalization(image)

        #converts image to black and white, inverts the colors, then stuffs the pixels into an ndarray
        image_matrix = self.standardize_image(image)/255
        #dilates or widens stroke thickness before resizing to 20x20 to prevent missing pixels
        stru = np.array([[1, 1], [1, 1]])
        image_matrix = ndimage.morphology.binary_dilation(image_matrix, stru, 5)
        #final resizing and stroke normalization
        width = 20
        height = 20
        image = ImageOps.invert(Image.fromarray(np.int8(image_matrix)*255).resize((width, height), Image.ANTIALIAS).convert('L'))
        image = self.stroke_normalization(image)
        return image
        
    def stroke_thickness(self, image_matrix):
        '''calculates the thickness of stroke in an image matrix'''
        #euclidean distance transformation
        image_edt = ndimage.distance_transform_edt(image_matrix)
        image_edt *= 10
        
        #medial axis transformation
        #image_skele = medial_axis(image_matrix)
        #image_skele = image_skele.astype(int)
        #image = Image.fromarray(image_skele*255)

        dist_values = [i for i in image_edt]
        thickness = np.mean(dist_values)
        return thickness

    def stroke_normalization(self, image):
        
        '''takes an image and checks if thickness is within a
            threshold, then uses erosion/dilation to adjust thickness
            appropriately'''   
        image_matrix = self.standardize_image(image)/255
        thickness = self.stroke_thickness(image_matrix)
        print thickness
        print image_matrix        
        while(abs(thickness-3.24267476061) > 0.6):
            #erodes/dialates binary image, then calculates the binary image thickness
            #converts np.bool type to int8 (data able to be handled by PIL)
            if(thickness > 3.24267476061):
                stru = np.array([[1,1],[1,1]])
                image_matrix = ndimage.morphology.binary_erosion(image_matrix, stru)
                thickness = self.stroke_thickness(Image.fromarray(np.int8(image_matrix)))
            else:
                stru = np.array([[1,1],[1,1]])
                image_matrix = ndimage.morphology.binary_dilation(image_matrix)
                thickness = self.stroke_thickness(Image.fromarray(np.int8(image_matrix)))
            print thickness
            print np.int8(image_matrix)
        image_matrix = ndimage.morphology.binary_closing(image_matrix)
        print np.int8(image_matrix)
        transformed_image = ImageOps.invert(Image.fromarray(np.int8(image_matrix*255)).convert('L'))
        return transformed_image

    def standardize_image(self, image):
        '''resizes image and turns it black and white'''
        image_c = image.convert('L')
        image_i = ImageOps.invert(image_c)
        #convert image to black and white
        image_b = image_i.point(lambda x: 0 if x<64 else 255, 'L')
        return np.array(image_b)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


    
three = Image.open("alex_3.png")
cow = Image.open("cow.jpg")
tu = Image.open("tulips.jpg")
two = Image.open("jeff_2.png")
chess = Image.open("chess.png")
sq = Image.open("ws.jpg")
eight = Image.open("anthony_8.png")
five = Image.open("andrew_5.png")
net = Network([784, 30, 10])
net.SGD(net.training_data, 5, 10, 3.0, test_data = net.test_data)
