#include <cstdio>
#include <math.h>

#define INPUT_SIZE 784
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define OUTPUT_SIZE 10
#define HIDDEN_SIZE 256

// 3-layer neural network
typedef struct {
    float *weights1;
    float *weights2;
    float *weights3;
    float *biases1;
    float *biases2;
    float *biases3;
} NeuralNetwork;

// checks if CUDA call was successful
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
            __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
}

void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fread(data, sizeof(float), size, file);

    fclose(file);
}

// He weight initialization
void weight_initialization(float *weights, int size) {
    float scale = sqrt(2.0 / HIDDEN_SIZE);
    for (int i = 0; i < size; i++) {
        weights[i] = scale * (2.0 * rand() / RAND_MAX - 1.0);
    } 
}

// bias initialization
void bias_initialization(float *biases, int size) {
    for (int i = 0; i < size; i++) {
        biases[i] = 0.0;
    }
}

void initialize_network(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights3, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->biases1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->biases2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->biases3, OUTPUT_SIZE * sizeof(float)));

    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_weights3 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias2 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias3 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // initialize weights and biases
    weight_initialization(h_weights1, INPUT_SIZE * HIDDEN_SIZE);
    weight_initialization(h_weights2, HIDDEN_SIZE * HIDDEN_SIZE);
    weight_initialization(h_weights3, HIDDEN_SIZE * OUTPUT_SIZE);
    bias_initialization(h_bias1, HIDDEN_SIZE);
    bias_initialization(h_bias2, HIDDEN_SIZE);
    bias_initialization(h_bias3, OUTPUT_SIZE);

    // copying weights and biases to device
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights3, h_weights3, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->biases1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->biases2, h_bias2, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->biases3, h_bias3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(h_weights1);
    free(h_weights2);
    free(h_weights3);
    free(h_bias1);
    free(h_bias2);
    free(h_bias3);
}

void train(NeuralNetwork *nn) {
    // training
    float *d_X_train, *d_X_test;
    int *d_y_train, *d_y_test;
}

int main() {
    NeuralNetwork nn;
    initialize_network(&nn);

    float *X_train = (float *)malloc(INPUT_SIZE * TRAIN_SIZE * sizeof(float));
    int *y_train = (int *)malloc(OUTPUT_SIZE * TRAIN_SIZE * sizeof(int));
    float *X_test = (float *)malloc(INPUT_SIZE * TEST_SIZE * sizeof(float));
    int *y_test = (int *)malloc(OUTPUT_SIZE * TEST_SIZE * sizeof(int));

    // loading train and test sets
    load_data("mnist_data/X_train.bin", X_train, INPUT_SIZE * TRAIN_SIZE);
    load_data("mnist_data/y_train.bin", (float *)y_train, TRAIN_SIZE);
    load_data("mnist_data/X_test.bin", X_test, INPUT_SIZE * TEST_SIZE);
    load_data("mnist_data/y_test.bin", (float *)y_test, TEST_SIZE);

    for (int k = 0; k < 5; k++) {
        for (int i = 0; i < sqrt(INPUT_SIZE); i++) {
            for (int j = 0; j < sqrt(INPUT_SIZE); j++) {
                if (X_train[k * INPUT_SIZE + i * (int)sqrt(INPUT_SIZE) + j] > 0) {
                    printf("x");
                } else {
                    printf(" ");
                }
            }
            printf("\n");
        }
        printf("This number is %d\n", y_train[k]);
    }

    train(&nn);
}