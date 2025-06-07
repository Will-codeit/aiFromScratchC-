#include <iostream>
#include <string>
#include <vector>
#include <cmath>  
#include <cstdlib>  
#include <ctime> 
#include <sstream>
#include <fstream>
#include <numeric>
#include <random>

//to save time
using namespace std;

//slight more efficent version of aiMNSIT
//still can be more efficent

// Neuron class 
class neuron { 
public:
    vector<double> weights;
    // Reserve capacity for weights in constructor
    vector<double> inputs;
    // Reserve capacity for inputs in constructor 
    double bias; 
    double output;
    double z; // Store pre-activation for backprop

    neuron(const size_t numInputs) {
        weights.reserve(numInputs);
        inputs.reserve(numInputs);
        for (size_t i = 0; i < numInputs; ++i) {
            //HE weight initialization works better with relu
            weights.push_back((double)rand() / RAND_MAX * sqrt(2.0 / numInputs)); // He initialization
        }

        bias = 0; //could randomly intialize bias also, but i removed that to try to avoid fading gradients
        output = 0.0;
        z = 0.0;
    }

    static double relu(double x) {
        return (x > 0) ? x : 0;
    }
    
    static double reluDerivative(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }
    //leaky relus avoid neurons dying.
    static double leakyRelu(double x) {
        return (x > 0) ? x : 0.01 * x; // Small slope for negative inputs
    }
    
    static double leakyReluDerivative(double x) {
        return (x > 0) ? 1.0 : 0.01; // Derivative of Leaky ReLU
    }

    double calculate(const vector<double>& inputVector) {
        if (inputVector.size() != weights.size()) {
            cerr << "Input size mismatch: " << inputVector.size() << " vs " << weights.size() << endl;
            return 0.0;
        }
        inputs = inputVector;
        z = inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0) + bias;
        output = relu(z);
        return output;
    }
};

// Layer class
class layer {
public:
//stores the neurons and their associated weights
    vector<neuron> neurons;
    vector<double> outputs;
    size_t layer_size;
    size_t input_size;
    //constructor
    layer(size_t num_neurons, size_t num_inputs) {
        layer_size = num_neurons;
        input_size = num_inputs;
        for (size_t i = 0; i < num_neurons; ++i) {
            neurons.push_back(neuron(num_inputs));
        }
        outputs.resize(num_neurons);
    }
//forward pass
    vector<double> forward(const vector<double>& inputs) {
        if (inputs.size() != input_size) {
            cerr << "Layer input size mismatch" << endl;
            return outputs;
        }
        for (size_t i = 0; i < neurons.size(); ++i) {
            outputs[i] = neurons[i].calculate(inputs);
        }
        return outputs;
    }
};

// Neural network class
class neuralNetwork {
    public:
    vector<layer> layers;
    //constructor,
    //layer count is number of hidden layers, output size is number of output neurons. 
        neuralNetwork(size_t layerCount, size_t inputSize, size_t outputSize) {
            //init - builds layers. 
            size_t currentInputSize = inputSize;
            

            if(layerCount >= 3){ 
                layers.push_back(layer(512, currentInputSize)); //3 hidden layers
                currentInputSize = 512;
            }
            if (layerCount >= 2){
                layers.push_back(layer(256, currentInputSize)); //2 hidden layer
                currentInputSize = 256;
            }
            if (layerCount >= 1){
                layers.push_back(layer(128, currentInputSize)); //1 hidden layer
                currentInputSize = 128;
            }
            layers.push_back(layer(10, currentInputSize)); // creates output layer with 10 neurons, 1 for each digit.
        }



        /*start of utility functions :
-----------------------------------------------------------------------------------------------------------------------------------------------------*/

        // Function to calculate the dot product of two vectors
    double dotProduct(const vector<double>& vec1, const vector<double>& vec2) {
            if (vec1.size() != vec2.size()) {
                cerr << "Vectors must have the same size" << endl;
            }
            return inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
        }

    void check_dead_neurons(const vector<vector<double>>& X) {
            cout << "\n=== Dead Neuron Analysis ===" << endl;

            for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
                vector<int> dead_count(layers[layer_idx].layer_size, 0);
                vector<int> total_activations(layers[layer_idx].layer_size, 0);

                // Test all training samples
                for (size_t sample = 0; sample < X.size(); ++sample) {
                    vector<double> currentOutput = X[sample];

                    // Forward pass up to this layer
                    for (size_t l = 0; l <= layer_idx; ++l) {
                        currentOutput = layers[l].forward(currentOutput);
                    }

                    // Check each neuron in this layer
                    for (size_t neuron_idx = 0; neuron_idx < layers[layer_idx].neurons.size(); ++neuron_idx) {
                        total_activations[neuron_idx]++;
                        //check if output is close to 0, if so count it as dead.
                        if (layers[layer_idx].neurons[neuron_idx].output < 1e-6) {
                            dead_count[neuron_idx]++;
                        }
                    }
                }

                // Calculate and report dead neuron statistics
                int completely_dead = 0;
                int mostly_dead = 0;
            
                //returns dead count for each layer
                cout << "Layer " << layer_idx << ":" << endl;
                for (size_t neuron_idx = 0; neuron_idx < layers[layer_idx].layer_size; ++neuron_idx) {
                    double dead_percentage = (double)dead_count[neuron_idx] / total_activations[neuron_idx] * 100.0;

                    if (dead_percentage == 100.0) {
                        completely_dead++;
                    } else if (dead_percentage > 90.0) {
                        mostly_dead++;
                    }

                    cout << "  Neuron " << neuron_idx << ": " << dead_percentage << "% dead ("
                         << dead_count[neuron_idx] << "/" << total_activations[neuron_idx] << ")" << endl;
                }

                cout << "  Summary: " << completely_dead << " completely dead, "
                     << mostly_dead << " mostly dead (>90%)" << endl;
                cout << "  Total neurons: " << layers[layer_idx].layer_size << endl;
                cout << endl;
            }
        }
        // Also add this method to check pre-activation values (z values)
    void check_preactivation_stats(const vector<vector<double>>& X) {
            cout << "\n=== Pre-activation (z) Statistics ===" << endl;

            for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
                //finds the minimum, maximum, and size of layers
                vector<double> min_z(layers[layer_idx].layer_size, INFINITY);
                vector<double> max_z(layers[layer_idx].layer_size, -INFINITY);
                vector<double> sum_z(layers[layer_idx].layer_size, 0.0);

                // Test all training samples
                for (size_t sample = 0; sample < X.size(); ++sample) {
                    vector<double> currentOutput = X[sample];

                    // Forward pass up to this layer
                    for (size_t l = 0; l <= layer_idx; ++l) {
                        currentOutput = layers[l].forward(currentOutput);
                    }

                    // Collect z statistics
                    for (size_t neuron_idx = 0; neuron_idx < layers[layer_idx].neurons.size(); ++neuron_idx) {
                        double z = layers[layer_idx].neurons[neuron_idx].z;
                        min_z[neuron_idx] = min(min_z[neuron_idx], z);
                        max_z[neuron_idx] = max(max_z[neuron_idx], z);
                        sum_z[neuron_idx] += z;
                    }
                }

                cout << "Layer " << layer_idx << " pre-activation stats:" << endl;
                for (size_t neuron_idx = 0; neuron_idx < layers[layer_idx].layer_size; ++neuron_idx) {
                    double avg_z = sum_z[neuron_idx] / X.size();
                    cout << "  Neuron " << neuron_idx << ": min=" << min_z[neuron_idx]
                         << ", max=" << max_z[neuron_idx] << ", avg=" << avg_z << endl;
                }
                cout << endl;
            }
        }

        // Softmax function for a vector of values
    static vector<double> softmax(const vector<double>& z) {
            vector<double> result(z.size());

            // Find the maximum value for numerical stability
            double max_val = *max_element(z.begin(), z.end());

            // Calculate exp(z_i - max) for each element
            double sum_exp = 0.0;
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] = exp(z[i] - max_val);  //prevents overflow, called log sum exp
                sum_exp += result[i];
            }

            // Normalize by dividing by the sum
            for (size_t i = 0; i < z.size(); ++i) {
                result[i] /= sum_exp;
            }

            return result;
        }

        // Softmax derivative for backpropagation
        // Returns the Jacobian matrix for a vector of outputs
    static vector<vector<double>> softmax_derivative(const vector<double>& softmax_output) {
            size_t n = softmax_output.size();
            vector<vector<double>> jacobian(n, vector<double>(n));

            for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                // Diagonal element: s_i * (1 - s_i)
                jacobian[i][j] = softmax_output[i] * (1.0 - softmax_output[i]);
                } else {
                // Off-diagonal element: -s_i * s_j
                jacobian[i][j] = -softmax_output[i] * softmax_output[j];
                }
            }
            }

            return jacobian;
        }
                
        // Cross-entropy loss functions:
 
        // Cross-entropy loss for multi-class classification (categorical)
    double categorical_cross_entropy(const vector<double>& predicted, const vector<double>& actual) {
            if (predicted.size() != actual.size()) {
                throw invalid_argument("Predicted and actual vectors must have the same size");
            }
            
            double loss = 0.0;
            double epsilon = 1e-15;
            
            for (size_t i = 0; i < predicted.size(); ++i) {
                // Clip predictions to prevent log(0)
                double clipped_pred = max(epsilon, min(1.0 - epsilon, predicted[i]));
                loss -= actual[i] * log(clipped_pred);
            }
            
            return loss;


        // Average cross-entropy loss over a batch
    }
    double average_categorical_cross_entropy(const vector<vector<double>>& predicted_batch, const vector<vector<double>>& actual_batch) {
            if (predicted_batch.size() != actual_batch.size()) {
                throw invalid_argument("Batch sizes must match");
            }
            
            double total_loss = 0.0;
            for (size_t i = 0; i < predicted_batch.size(); ++i) {
                total_loss += categorical_cross_entropy(predicted_batch[i], actual_batch[i]);
            }
            
            return total_loss / predicted_batch.size();
        }
   

        // Cross-entropy derivative for backpropagation (when used with softmax)
        // This is the simplified version: predicted - actual
    vector<double> cross_entropy_softmax_derivative(const vector<double>& predicted, const vector<double>& actual) {
            if (predicted.size() != actual.size()) {
                throw invalid_argument("Predicted and actual vectors must have the same size");
            }
            
            vector<double> derivative(predicted.size());
            for (size_t i = 0; i < predicted.size(); ++i) {
                derivative[i] = predicted[i] - actual[i];
            }
            
            return derivative;
        }

        // Alternative: Simplified softmax derivative for cross-entropy loss
        // When using cross-entropy loss, the derivative simplifies to (y_pred - y_true)
    static vector<double> softmax_cross_entropy_derivative(const vector<double>& predicted, const vector<double>& actual) {
            vector<double> derivative(predicted.size());

            for (size_t i = 0; i < predicted.size(); ++i) {
                derivative[i] = predicted[i] - actual[i];
            }

            return derivative;
        }

    void normalize_data(const vector<vector<uint8_t>>& x, const vector<uint8_t>& y, vector<vector<double>>& xTrainDouble, vector<double>& yTrainDouble) {
            // Normalize the input data (x) from uint8_t to double, scaling values to be between 0 and 1.
            vector<vector<double>> normalized_x(x.size(), vector<double>(x[0].size()));
            for (size_t i = 0; i < x.size(); ++i) {
                for (size_t j = 0; j < x[i].size(); ++j) {
                    normalized_x[i][j] = static_cast<double>(x[i][j]) / 255.0; // Normalize pixel values
                }
            }
    
            // Convert labels (y) from uint8_t to double for compatibility with the neural network.
            vector<double> normalized_y(y.size());
            for (size_t i = 0; i < y.size(); ++i) {
                normalized_y[i] = static_cast<double>(y[i]);
            }
    
            // Update the output references with normalized data.
            xTrainDouble = normalized_x;
            yTrainDouble = normalized_y;
        }
    
    void one_hot_encode(const vector<double>& y, vector<vector<double>>& y_encoded, size_t num_classes) { //changes actual output to vector - so it has same shape as predictions.
            y_encoded.resize(y.size(), vector<double>(num_classes, 0.0)); //num_clases would be 10
            for (size_t i = 0; i < y.size(); ++i) {
                y_encoded[i][static_cast<size_t>(y[i])] = 1.0; // Set the appropriate class index to 1.0
            }
        }
        /*end of utility functions
-----------------------------------------------------------------------------------------------------------------------------------------------------*/

    void trainBatch(vector<vector<double>> &X, vector<double> &Y, double lr, size_t epochs) {
        //prepare random
        random_device rd;  
        mt19937 g(rd());  
        // Create a vector of indices
        vector<size_t> indices(X.size());
        iota(indices.begin(), indices.end(), 0);


        // Iterate through epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            lr = lr * (1.0 / (1.0 + 0.05 * epoch)); // Learning rate decay
            // In-place shuffle (more memory efficient)  

            // Shuffle indices at the start of each epoch  
            shuffle(indices.begin(), indices.end(), g);
            for (size_t i = 0; i < indices.size(); ++i) {  

                if (i != indices[i]) {  
                    swap(X[i], X[indices[i]]);  
                    swap(Y[i], Y[indices[i]]);  
                    // Update indices array to reflect the swap  
                    for (size_t j = i + 1; j < indices.size(); ++j) {  
                        if (indices[j] == i) indices[j] = indices[i];  
                    }  
                }       
            }  
            // Implement mini-batch gradient descent
            size_t batch_size = 128;
            int batch_count = 0;
            // Iterate through X, jumping by batch size each time
            for (size_t batch_start = 0; batch_start < X.size(); batch_start += batch_size) {
                double batchLoss = 0.0;
                
                // Initialize gradient accumulators for this batch
                vector<vector<vector<double>>> weight_gradients(layers.size());
                vector<vector<double>> bias_gradients(layers.size());
                
                for (size_t l = 0; l < layers.size(); ++l) {
                    weight_gradients[l].resize(layers[l].neurons.size());
                    bias_gradients[l].resize(layers[l].neurons.size(), 0.0);
                    
                    for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                        weight_gradients[l][j].resize(layers[l].neurons[j].weights.size(), 0.0);
                    }
                }
                
                size_t actual_batch_size = 0;


                for (size_t i = batch_start; i < min(batch_start + batch_size, X.size()); ++i) {
                    actual_batch_size++;
                    
                    // Forward pass for this sample
                    vector<vector<double>> layerOutputs;
                    vector<double> currentOutput = X[i];
                    layerOutputs.push_back(currentOutput); // Input layer
                    
                    for (auto &layer : layers) {
                        currentOutput = layer.forward(currentOutput);
                        layerOutputs.push_back(currentOutput);
                    }
                    
                    // Apply softmax to the output layer
                    vector<double> predictedOutput = softmax(currentOutput);
                    
                    // One-hot encode the target for this sample
                    vector<vector<double>> actualOutput;
                    vector<double> single_label = {Y[i]};
                    one_hot_encode(single_label, actualOutput, 10);
                    
                    // Calculate loss for this sample
                    batchLoss += categorical_cross_entropy(predictedOutput, actualOutput[0]);
                    
                    // Initialize deltas for backpropagation
                    vector<vector<double>> deltas(layers.size());
                    for (size_t l = 0; l < layers.size(); ++l) {
                        deltas[l].resize(layers[l].layer_size);
                    }
                    
                    // Calculate the derivative of the loss with respect to the output layer
                    deltas.back() = cross_entropy_softmax_derivative(predictedOutput, actualOutput[0]);
                    
                    // Backpropagation for this sample
                    for (int l = layers.size() - 1; l >= 0; --l) {
                        for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                            double error = 0.0;
                            if (l == layers.size() - 1) {
                                // Output layer error
                                error = deltas[l][j];
                            } else {
                                // Hidden layer error
                                for (size_t k = 0; k < layers[l + 1].neurons.size(); ++k) {
                                    error += deltas[l + 1][k] * layers[l + 1].neurons[k].weights[j];
                                }
                            }
                            deltas[l][j] = error * neuron::reluDerivative(layers[l].neurons[j].z);
                        }
                    }
                    
                    // Accumulate gradients for this sample
                    for (size_t l = 0; l < layers.size(); ++l) {
                        for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                            // Accumulate weight gradients
                            for (size_t k = 0; k < layers[l].neurons[j].weights.size(); ++k) {
                                weight_gradients[l][j][k] += deltas[l][j] * layerOutputs[l][k];
                            }
                            // Accumulate bias gradients
                            bias_gradients[l][j] += deltas[l][j];
                        }
                    }
                }
                
                // Update weights and biases using averaged gradients
                for (size_t l = 0; l < layers.size(); ++l) {
                    for (size_t j = 0; j < layers[l].neurons.size(); ++j) {
                        // Update weights with averaged gradients
                        for (size_t k = 0; k < layers[l].neurons[j].weights.size(); ++k) {
                            layers[l].neurons[j].weights[k] -= lr * (weight_gradients[l][j][k] / actual_batch_size);
                        }
                        // Update bias with averaged gradients
                        layers[l].neurons[j].bias -= lr * (bias_gradients[l][j] / actual_batch_size);
                    }
                }
                batch_count++;

                // Print loss and batch count so we can track how its doing
                if (batch_count % 50 == 0){
                cout << "Epoch " << epoch << " | Batch " << batch_count << " | Loss " << batchLoss / actual_batch_size << endl;
                }
            }
        }
    }
};



// Test class
class Test {
public:
    // Function to evaluate the neural network on test data
    void evaluate(neuralNetwork& nn, const vector<vector<double>>& x_test, const vector<double>& y_test) {
        if (x_test.size() != y_test.size()) {
            cerr << "Error: x_test and y_test must have the same number of samples." << endl;
            return;
        }

        size_t correct_predictions = 0;
        for (size_t i = 0; i < x_test.size(); ++i) {
            vector<double> output = x_test[i];
            for (auto& layer : nn.layers) {
                output = layer.forward(output);
            }

            // Apply softmax to the output layer
            vector<double> softmax_output = neuralNetwork::softmax(output);

            // Find the predicted class
            size_t predicted_class = distance(softmax_output.begin(), max_element(softmax_output.begin(), softmax_output.end()));

            // Check if the prediction matches the actual label
            if (predicted_class == static_cast<size_t>(y_test[i])) {
                correct_predictions++;
            }
        }

        double accuracy = static_cast<double>(correct_predictions) / x_test.size() * 100.0;
        cout << "Accuracy: " << accuracy << "%" << endl;
    }
};




    //data loading
    //premade loader from mnist site
class MnistDataloader {  
public:  
    string training_images_filepath;  
    string training_labels_filepath;  
    string test_images_filepath;  
    string test_labels_filepath;  
  
    MnistDataloader(const string& train_img, const string& train_lbl,  
                    const string& test_img, const string& test_lbl)  
        : training_images_filepath(train_img), training_labels_filepath(train_lbl),  
          test_images_filepath(test_img), test_labels_filepath(test_lbl) {}  
  
    void read_images_labels(const string& images_filepath, const string& labels_filepath,  
                            vector<vector<uint8_t>>& images, vector<uint8_t>& labels) {  
        ifstream label_file(labels_filepath, ios::binary);  
        if (!label_file.is_open()) throw runtime_error("Cannot open label file");  
        uint32_t magic = 0, size = 0;  
        label_file.read((char*)&magic, 4);  
        label_file.read((char*)&size, 4);  
        magic = __builtin_bswap32(magic);  
        size = __builtin_bswap32(size);  
        if (magic != 2049) throw runtime_error("Label file magic number mismatch");  
        labels.resize(size);  
        label_file.read((char*)labels.data(), size);  
  
        ifstream image_file(images_filepath, ios::binary);  
        if (!image_file.is_open()) throw runtime_error("Cannot open image file");  
        uint32_t img_magic = 0, img_size = 0, rows = 0, cols = 0;  
        image_file.read((char*)&img_magic, 4);  
        image_file.read((char*)&img_size, 4);  
        image_file.read((char*)&rows, 4);  
        image_file.read((char*)&cols, 4);  
        img_magic = __builtin_bswap32(img_magic);  
        img_size = __builtin_bswap32(img_size);  
        rows = __builtin_bswap32(rows);  
        cols = __builtin_bswap32(cols);  
        if (img_magic != 2051) throw runtime_error("Image file magic number mismatch");  
        images.resize(img_size, vector<uint8_t>(rows * cols));  
        for (uint32_t i = 0; i < img_size; ++i) {  
            image_file.read((char*)images[i].data(), rows * cols);  
        }  
    }  
  
    void load_data(vector<vector<uint8_t>>& x_train, vector<uint8_t>& y_train,  
                   vector<vector<uint8_t>>& x_test, vector<uint8_t>& y_test) {  
        read_images_labels(training_images_filepath, training_labels_filepath, x_train, y_train);  
        read_images_labels(test_images_filepath, test_labels_filepath, x_test, y_test);  
    }  
};  

int main() {

    //the paths of the files
    string input_path = "/Users/will/Downloads/archive";  
    string training_images_filepath = input_path + "/train-images-idx3-ubyte/train-images-idx3-ubyte";  
    string training_labels_filepath = input_path + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte";  
    string test_images_filepath = input_path + "/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";  
    string test_labels_filepath = input_path + "/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";  
  

    cout << "Loading data.." << endl << endl;

    //load the data using the dataloader from the site i got the mnist data from
    MnistDataloader mnist(training_images_filepath, training_labels_filepath,  
                          test_images_filepath, test_labels_filepath);  
  
    vector<vector<uint8_t>> x_train, x_test;  
    vector<uint8_t> y_train, y_test;  
    mnist.load_data(x_train, y_train, x_test, y_test);  
        cout << "training data size : " << (x_train[0].size()) << endl;
    if (x_train.size() != y_train.size()) {
        cerr << "Error: x_train and y_train must have the same number of samples." << endl;
        return -1;
    }
    
    
    
    
    //create neural network

    cout << "Creating neural network.." << endl << endl;

    size_t s = x_train[0].size();
    neuralNetwork nn(1, s, 10);
    
    cout << "Training model... take a quick nap" << endl << endl;
    

    //change training data to double, and normalize.
    vector<vector<double>> xTrainDouble;
    vector<double> yTrainDouble;
    nn.normalize_data(x_train, y_train, xTrainDouble, yTrainDouble);

    //train (in batches) on the train data, then evaluate
    nn.trainBatch(xTrainDouble, yTrainDouble, 0.01, 15);

    //convert and normalize test data
    vector<vector<double>> xTestDouble;
    vector<double> yTestDouble;
    nn.normalize_data(x_test, y_test, xTestDouble, yTestDouble);

    //test using test data
    Test t;
    t.evaluate(nn, xTestDouble, yTestDouble);
    cout << "Completed training!" << endl;
    

}
