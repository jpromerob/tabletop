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
#include <stdbool.h>

#define COLOR_HIGH 1
#define K_SZ 29
#define NB_NETS 4

#define WINDOW_WIDTH 280
#define WINDOW_SLICE 181
#define WINDOW_HEIGHT WINDOW_SLICE*NB_NETS
#define NB_FRAMES 5000

#define BUFFER_SIZE 1024 * 256
#define PORT_UDP_RAW 3330
#define PORT_UDP_CNN 3331

int emptyMatrix[WINDOW_WIDTH][WINDOW_HEIGHT];

int input_raw_mat[WINDOW_WIDTH][WINDOW_HEIGHT]; // where incoming raw data is stored
int input_cnn_mat[WINDOW_WIDTH][WINDOW_HEIGHT]; // where incoming scnn data is stored
int shared_raw_mat[WINDOW_WIDTH][WINDOW_HEIGHT]; // matrix shared between udp receiver (raw) and render|saver
int shared_cnn_mat[WINDOW_WIDTH][WINDOW_HEIGHT]; // matrix shared between udp receiver (scnn) and render|saver
int visual_raw_mat[WINDOW_WIDTH][WINDOW_HEIGHT]; // matrix used by render (raw)
int visual_cnn_mat[WINDOW_WIDTH][WINDOW_HEIGHT]; // matrix used by render (scnn)

float acc_time = 0.001000;
int x_px[NB_NETS];
int y_px[NB_NETS];
int scale = 1;
bool is_live = false;
bool is_the_end = false;

int base = 0xFF000000;
int black = 0x00000000;
int green = 0x00006400;
int blue = 0x000000FF;
int red = 0x00FF0000;
int yellow = 0x00FFFF00;

pthread_mutex_t xyp_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t end_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t raw_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t cnn_mutex = PTHREAD_MUTEX_INITIALIZER;

// Function to calculate Euclidean distance between two points
double get_distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

void* updateRaw(void* arg) {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[BUFFER_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    } else {
        printf("Socket for RAW correctly created\n");
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

    printf("Listening for RAW data...\n");

    struct timespec start_time, current_time;
    bool local_end = false;

    while (!local_end) {

        
        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  
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

                for (int j = 0; j < NB_NETS; j++) {
                    input_raw_mat[x][y+WINDOW_SLICE*j] += 1;
                }
            }
            
            // Get the current time
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            
            // Calculate the elapsed time in seconds
            elapsed_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            
        } while (elapsed_time < acc_time); // Repeat until 1ms has elapsed
        
        // Acquire mutex lock | update data | release mutex lock
        pthread_mutex_lock(&raw_mutex);
        memcpy(shared_raw_mat, input_raw_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&raw_mutex);

        memcpy(input_raw_mat, emptyMatrix, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);

    }

    close(sockfd);

    return NULL;
}


void* updateCnn(void* arg) {
    
    int screen_id = *((int *)arg);
    printf("Screen ID: %d\n", screen_id);

    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[BUFFER_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    } else {
        printf("Socket for CNN correctly created\n");
    }

    int port_cnn = PORT_UDP_CNN+screen_id*3;

    // Initialize server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_cnn); // Use the same port as in Python

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for CNN data on port %d...\n", port_cnn);

    struct timespec start_time, current_time;
    bool local_end = false;

    while (!local_end) {

        
        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  
    
        
        int recv_counter = 0;
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

            input_cnn_mat[x][y+WINDOW_SLICE*screen_id] += 1;
        }
            
        int sum_idx_x = 0;
        int sum_idx_y = 0;
        int count_ones = 0;
        for (int y = WINDOW_SLICE*(screen_id+0); y < WINDOW_SLICE*(screen_id+1); y++) {
            for (int x = 0; x < WINDOW_WIDTH; x++) {
                if(visual_cnn_mat[x][y]>0){
                    sum_idx_x+=x;
                    sum_idx_y+=y;
                    count_ones++;
                }
            }
        }
        
        if(count_ones > 0) {
            pthread_mutex_lock(&xyp_mutex);  
            x_px[screen_id] = sum_idx_x/count_ones;
            y_px[screen_id] = sum_idx_y/count_ones;
            pthread_mutex_unlock(&xyp_mutex);  
        }        

    }

    close(sockfd);

    return NULL;
}


void* updateShared(void* arg) {

    bool local_end = false;

    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  

        usleep(1000000*acc_time);

        pthread_mutex_lock(&cnn_mutex);        
        memcpy(shared_cnn_mat, input_cnn_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&cnn_mutex);

        memcpy(input_cnn_mat, emptyMatrix, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    }
}


// Function to render the entire matrix as a texture
void* renderMatrix(void* arg) {

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Air Hockey", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, scale*WINDOW_WIDTH, scale*WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* matrixTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, WINDOW_HEIGHT);


    bool local_end = false;

    while (!local_end) {

        
        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  
        
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
        memcpy(visual_raw_mat, shared_raw_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&raw_mutex);

        pthread_mutex_lock(&cnn_mutex);
        memcpy(visual_cnn_mat, shared_cnn_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&cnn_mutex);     


        int local_x_px[NB_NETS];
        int local_y_px[NB_NETS];
        pthread_mutex_lock(&xyp_mutex);          
        memcpy(local_x_px, x_px, sizeof(int) * NB_NETS);
        memcpy(local_y_px, y_px, sizeof(int) * NB_NETS);
        pthread_mutex_unlock(&xyp_mutex);  

        // Lock the texture for writing
        void* pixels;
        int pitch;
        SDL_LockTexture(matrixTexture, NULL, &pixels, &pitch);

        int color = black;

        // Update the texture pixel data
        for (int screen_id = 0; screen_id < NB_NETS; screen_id++) {
            for (int y = WINDOW_SLICE*(screen_id+0); y < WINDOW_SLICE*(screen_id+1); y++) {
                for (int x = 0; x < WINDOW_WIDTH; x++) {
                    int index = y * WINDOW_WIDTH + x;
                    color = black;
                    if (visual_raw_mat[x][y] >= 1) {
                        // Set color to green (0xFF00FF00) if the matrix value is 1
                        color = green*(visual_raw_mat[x][y]/COLOR_HIGH);
                    } 
                    if (get_distance(x, y, local_x_px[screen_id], local_y_px[screen_id])<=3){
                        color = yellow;
                    }
                    if (visual_cnn_mat[x][y] >= 1) {
                        // Set color to green (0xFF00FF00) if the matrix value is 1
                        color = red*(visual_cnn_mat[x][y]/COLOR_HIGH);
                    } 
                    ((Uint32*)pixels)[index] = color + base;
                }
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
        SDL_Delay(1); // Adjust the delay as needed for your desired frame rate
    }
}



int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <scale> <acc_time_ms>\n", argv[0]);
        return 1;
    }

    char *operation = argv[1];
    scale = atoi(argv[1]);
    acc_time = (float)(atoi(argv[2]))/1000;

    int screen_id[NB_NETS];
    for (int i = 0; i < NB_NETS; i++) {
        screen_id[i] = i;
        x_px[i] = WINDOW_WIDTH/2;
        y_px[i] = WINDOW_SLICE*(i+0.5);
    }

    memset(input_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(shared_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(emptyMatrix, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);

    pthread_t updateRawThread, updateSharedThread, updateCnnThreads[NB_NETS];
    pthread_t renderThread;


    // Create threads
    pthread_create(&updateRawThread, NULL, updateRaw, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_create(&updateCnnThreads[i], NULL, updateCnn, (void *)&screen_id[i]);
    }
    pthread_create(&updateSharedThread, NULL, updateShared, NULL);
    pthread_create(&renderThread, NULL, renderMatrix, NULL);

    // Wait for threads to finish (this will never happen in this example)
    pthread_join(updateRawThread, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_join(updateCnnThreads[i], NULL);
    }
    pthread_join(updateSharedThread, NULL);
    pthread_join(renderThread, NULL);

    // Clean up mutex
    pthread_mutex_destroy(&raw_mutex);
    pthread_mutex_destroy(&cnn_mutex);

    return 0;
}
