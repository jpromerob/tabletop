#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <SDL.h>
#include <SDL2/SDL.h>
#include <arpa/inet.h>
#include <time.h>
#include <math.h>

#define ACC_TIME 0.002

#define WINDOW_WIDTH 280
#define WINDOW_HEIGHT 181
#define K_SZ 29

#define NB_COLS 280
#define NB_ROWS 181

#define BUFFER_SIZE 1024 * 256
#define PORT_UDP_RAW 3330
#define PORT_UDP_CNN 3331

int input_raw_mat[NB_COLS][NB_ROWS];
int output_raw_mat[NB_COLS][NB_ROWS];
int shared_raw_mat[NB_COLS][NB_ROWS];
int input_cnn_mat[NB_COLS][NB_ROWS];
int output_cnn_mat[NB_COLS][NB_ROWS];
int shared_cnn_mat[NB_COLS][NB_ROWS];
int emptyMatrix[NB_COLS][NB_ROWS];
int scale = 1;

pthread_mutex_t raw_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t cnn_mutex = PTHREAD_MUTEX_INITIALIZER;

// Function to calculate Euclidean distance between two points
double get_distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

void* updateRawMat(void* arg) {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[BUFFER_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    } else {
        printf("Socket correctly created\n");
    }

    // Initialize server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT_UDP_RAW); // Use the same port as in Python

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for data...\n");

    struct timespec start_time, current_time;

    while (1) {
        double elapsed_time;
    
        // Get the current time
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        int recv_counter = 0;
        // Perform the operation repeatedly until 1ms (0.001 seconds) has elapsed
        do {
            ssize_t recv_len = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                                    (struct sockaddr *)&client_addr, &client_addr_len);

            if (recv_len < 0) {
                perror("recvfrom");
                exit(1);
            } else {
                recv_counter++;
            }

            // Assuming the received data is packed as little-endian 32-bit integers
            for (int i = 0; i < recv_len; i += 4) {
                unsigned int packed_data;
                memcpy(&packed_data, &buffer[i], 4);

                // Extract x and y values
                int x = (packed_data >> 16) & 0x00003FFF;
                int y = (packed_data >> 0) & 0x00003FFF;

                input_raw_mat[x][y] = 1;
            }
            
            // Get the current time
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            
            // Calculate the elapsed time in seconds
            elapsed_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            
        } while (elapsed_time < ACC_TIME); // Repeat until 1ms has elapsed
        
        // Acquire mutex lock | update data | release mutex lock
        pthread_mutex_lock(&raw_mutex);        
        memcpy(shared_raw_mat, input_raw_mat, sizeof(int) * NB_COLS * NB_ROWS);
        pthread_mutex_unlock(&raw_mutex);

        memcpy(input_raw_mat, emptyMatrix, sizeof(int) * NB_COLS * NB_ROWS);

    }

    close(sockfd);

    return NULL;
}

void* updateCnnMat(void* arg) {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[BUFFER_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    } else {
        printf("Socket correctly created\n");
    }

    // Initialize server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT_UDP_CNN); // Use the same port as in Python

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for data...\n");

    struct timespec start_time, current_time;

    while (1) {
        double elapsed_time;
    
        // Get the current time
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        int recv_counter = 0;
        // Perform the operation repeatedly until 1ms (0.001 seconds) has elapsed
        do {
            ssize_t recv_len = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                                    (struct sockaddr *)&client_addr, &client_addr_len);

            if (recv_len < 0) {
                perror("recvfrom");
                exit(1);
            } else {
                recv_counter++;
            }

            // Assuming the received data is packed as little-endian 32-bit integers
            for (int i = 0; i < recv_len; i += 4) {
                unsigned int packed_data;
                memcpy(&packed_data, &buffer[i], 4);

                // Extract x and y values
                int x = (K_SZ/2)+ ((packed_data >> 16) & 0x00003FFF);
                int y = (K_SZ/2)+ ((packed_data >> 0) & 0x00003FFF);

                input_cnn_mat[x][y] = 1;
            }
            
            // Get the current time
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            
            // Calculate the elapsed time in seconds
            elapsed_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            
        } while (elapsed_time < ACC_TIME); // Repeat until 1ms has elapsed
        
        // Acquire mutex lock | update data | release mutex lock
        pthread_mutex_lock(&cnn_mutex);        
        memcpy(shared_cnn_mat, input_cnn_mat, sizeof(int) * NB_COLS * NB_ROWS);
        pthread_mutex_unlock(&cnn_mutex);

        memcpy(input_cnn_mat, emptyMatrix, sizeof(int) * NB_COLS * NB_ROWS);

    }

    close(sockfd);

    return NULL;
}
// Function to render the entire matrix as a texture
void* renderMatrix(void* arg) {


    int black = 0xFF000000;
    int green = 0xFF00FF00;
    int blue = 0xFF0000FF;
    int red = 0xFFFF0000;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Air Hockey", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, scale*WINDOW_WIDTH, scale*WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* matrixTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, NB_COLS, NB_ROWS);

    while (1) {
        SDL_Event e;
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                // Cleanup and quit SDL
                SDL_DestroyRenderer(renderer);
                SDL_DestroyWindow(window);
                SDL_Quit();
                exit(0);
            }
        }

        // Acquire mutex lock | get data | release mutex lock
        pthread_mutex_lock(&raw_mutex);
        memcpy(output_raw_mat, shared_raw_mat, sizeof(int) * NB_COLS * NB_ROWS);
        pthread_mutex_unlock(&raw_mutex);
        pthread_mutex_lock(&cnn_mutex);
        memcpy(output_cnn_mat, shared_cnn_mat, sizeof(int) * NB_COLS * NB_ROWS);
        pthread_mutex_unlock(&cnn_mutex);



        int sum_idx_x = 0;
        int sum_idx_y = 0;
        int count_ones = 0;
        int avg_idx_x = 0;
        int avg_idx_y = 0;
        for (int y = 0; y < NB_ROWS; y++) {
            for (int x = 0; x < NB_COLS; x++) {
                if(output_cnn_mat[x][y]>0){
                    sum_idx_x+=x;
                    sum_idx_y+=y;
                    count_ones++;
                }
            }
        }
        
        if(count_ones > 0) {
            avg_idx_x = sum_idx_x/count_ones;
            avg_idx_y = sum_idx_y/count_ones;
        }


        // Lock the texture for writing
        void* pixels;
        int pitch;
        SDL_LockTexture(matrixTexture, NULL, &pixels, &pitch);

        int color = black;

        // Update the texture pixel data
        for (int y = 0; y < NB_ROWS; y++) {
            for (int x = 0; x < NB_COLS; x++) {
                int index = y * NB_COLS + x;
                color = black;
                if (output_raw_mat[x][y] == 1) {
                    // Set color to green (0xFF00FF00) if the matrix value is 1
                    color = green;
                } 
                if (output_cnn_mat[x][y] == 1) {
                    // Set color to green (0xFF00FF00) if the matrix value is 1
                    color = blue;
                } 
                if (get_distance(x, y, avg_idx_x, avg_idx_y)<4){
                    color = red;
                }
                ((Uint32*)pixels)[index] = color;
            }
        }

        // Unlock the texture
        SDL_UnlockTexture(matrixTexture);

        // Clear the renderer
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Render the texture
        SDL_RenderCopy(renderer, matrixTexture, NULL, NULL);

        // Update the screen
        SDL_RenderPresent(renderer);

        // Delay to limit the frame rate
        SDL_Delay(16); // Adjust the delay as needed for your desired frame rate
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("Usage: %s <scale>\n", argv[0]);
        return 1;
    }

    // Convert the command-line arguments (strings) to integers
    scale = atoi(argv[1]);

    memset(input_raw_mat, 0, sizeof(int) * NB_COLS * NB_ROWS);
    memset(shared_raw_mat, 0, sizeof(int) * NB_COLS * NB_ROWS);
    memset(emptyMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);

    pthread_t updateRawThread, updateCnnThread, renderThread;

    // Create threads
    pthread_create(&updateRawThread, NULL, updateRawMat, NULL);
    pthread_create(&updateCnnThread, NULL, updateCnnMat, NULL);
    pthread_create(&renderThread, NULL, renderMatrix, NULL);

    // Wait for threads to finish (this will never happen in this example)
    pthread_join(updateRawThread, NULL);
    pthread_join(updateCnnThread, NULL);
    pthread_join(renderThread, NULL);

    // Clean up mutex
    pthread_mutex_destroy(&raw_mutex);
    pthread_mutex_destroy(&cnn_mutex);

    return 0;
}
