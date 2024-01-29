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
#define NB_NETS 3
#define NB_SLICES (NB_NETS + 1)
#define EV_COUNT_THRESHOLD 2

#define WINDOW_WIDTH 280
#define SLICE_HEIGHT 181
#define WINDOW_HEIGHT SLICE_HEIGHT*NB_SLICES
#define NB_FRAMES 5000

#define BUFFER_SIZE 1024 * 256
#define PORT_UDP_RAW 3330
#define PORT_UDP_CNN 3331

int emptyMatrix[WINDOW_HEIGHT][WINDOW_WIDTH];

int input_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // where incoming raw data is stored
int input_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // where incoming scnn data is stored
int shared_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix shared between udp receiver (raw) and render|saver
int shared_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix shared between udp receiver (scnn) and render|saver
int visual_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used by render (raw)
int visual_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used by render (scnn)

float acc_time = 0.001000;
int x_px[NB_SLICES];
int y_px[NB_SLICES];
int ev_count[NB_SLICES];
float scale = 1.0;
bool is_live = false;
bool is_the_end = false;

int base = 0xFF000000;
int black = 0x00000000;
int green = 0x00006400;
int blue = 0x000000FF;
int cyan = 0x0000FFFF;
int magenta = 0x00FF66FF;
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

                for (int j = 0; j < NB_SLICES; j++) {
                    input_raw_mat[y+SLICE_HEIGHT*j][x] += 1;
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
        double elapsed_time;
    
        // Get the current time
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    
        
        int recv_counter = 0;

        int sum_idx_x = 0;
        int sum_idx_y = 0;
        int count_ones = 0;

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

                input_cnn_mat[y+SLICE_HEIGHT*screen_id][x] += 1;
            }
            
            for (int y = SLICE_HEIGHT*(screen_id+0); y < SLICE_HEIGHT*(screen_id+1); y++) {
                for (int x = 0; x < WINDOW_WIDTH; x++) {
                    if(visual_cnn_mat[y][x]>0){
                        sum_idx_x+=x;
                        sum_idx_y+=y;
                        count_ones++;
                    }
                }
            } 
            // Get the current time
            clock_gettime(CLOCK_MONOTONIC, &current_time);

            // Calculate the elapsed time in seconds
            elapsed_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
        } while (elapsed_time < acc_time); // Repeat until 1ms has elapsed

        
        if(count_ones > 0) {
            pthread_mutex_lock(&xyp_mutex);  
            x_px[screen_id] = sum_idx_x/count_ones;
            y_px[screen_id] = sum_idx_y/count_ones;
            ev_count[screen_id] = count_ones;
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
    int local_x_px[NB_SLICES];
    int local_y_px[NB_SLICES];
    int local_ev_count[NB_SLICES];
    int mux_id = NB_SLICES-1;

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


        pthread_mutex_lock(&xyp_mutex);          
        memcpy(local_x_px, x_px, sizeof(int) * NB_SLICES);
        memcpy(local_y_px, y_px, sizeof(int) * NB_SLICES);
        memcpy(local_ev_count, ev_count, sizeof(int) * NB_SLICES);
        pthread_mutex_unlock(&xyp_mutex);  


        
        mux_id = NB_NETS-1;
        for(int screen_id = 0; screen_id < NB_NETS; screen_id++){
            if(local_ev_count[screen_id] > EV_COUNT_THRESHOLD){
                mux_id = screen_id;
                // printf("ev count for ID: %d = %d\n", screen_id, local_ev_count[screen_id]);
                break;
            }
        }

        memcpy(visual_cnn_mat[SLICE_HEIGHT*(NB_SLICES-1)], shared_cnn_mat[SLICE_HEIGHT*mux_id], sizeof(int) * WINDOW_WIDTH * SLICE_HEIGHT);
        local_x_px[NB_SLICES-1] = local_x_px[mux_id];
        local_y_px[NB_SLICES-1] = local_y_px[mux_id]-mux_id*SLICE_HEIGHT+NB_NETS*SLICE_HEIGHT;
               
        int mux_color;
        if(mux_id == 0){
            mux_color = cyan;
            // printf("Mux ID: %d | Mux Color: cyan | (%d, %d)\n", mux_id, local_x_px[NB_SLICES-1], local_y_px[NB_SLICES-1]);
        } else if (mux_id == 1) {
            mux_color = magenta;
            // printf("Mux ID: %d | Mux Color: magenta | (%d, %d)\n", mux_id, local_x_px[NB_SLICES-1], local_y_px[NB_SLICES-1]);
        } else {
            mux_color = yellow;
            // printf("Mux ID: %d | Mux Color: yellow | (%d, %d)\n", mux_id, local_x_px[NB_SLICES-1], local_y_px[NB_SLICES-1]);
        }

        // printf("Mux ID: %d --> X[%d | %d | %d | %d]\n", mux_id, local_x_px[0], local_x_px[1], local_x_px[2], local_x_px[3]);
        // printf("Mux ID: %d --> Y[%d | %d | %d | %d]\n", mux_id, local_y_px[0], local_y_px[1], local_y_px[2], local_y_px[3]);
        // printf("Mux ID: %d --> [%d | %d | %d | %d]\n", mux_id, local_ev_count[0], local_ev_count[1], local_ev_count[2], local_ev_count[3]);

        // Lock the texture for writing
        void* pixels;
        int pitch;
        SDL_LockTexture(matrixTexture, NULL, &pixels, &pitch);

        int color = black;

        // Update the texture pixel data
        for (int screen_id = 0; screen_id < NB_SLICES; screen_id++) {
            // printf("Limits: %d and %d\n", SLICE_HEIGHT*(screen_id+0), SLICE_HEIGHT*(screen_id+1));
            for (int y = SLICE_HEIGHT*(screen_id+0); y < SLICE_HEIGHT*(screen_id+1); y++) {
                for (int x = 0; x < WINDOW_WIDTH; x++) {
                    int index = y * WINDOW_WIDTH + x;
                    color = black;
                    if (visual_raw_mat[y][x] >= 1) {
                        // Set color to green (0xFF00FF00) if the matrix value is 1
                        color = green*(visual_raw_mat[y][x]/COLOR_HIGH);
                    } 
                    if (get_distance(x, y, local_x_px[screen_id], local_y_px[screen_id])<=3){
                        if(screen_id == 0){
                            color = cyan;
                        } else if (screen_id == 1) {
                            color = magenta;
                        } else if (screen_id == 2){
                            color = yellow;
                        } else {
                            color = mux_color;
                        }
                    }
                    if (visual_cnn_mat[y][x] >= 1) {
                        // Set color to green (0xFF00FF00) if the matrix value is 1
                        color = red*(visual_cnn_mat[y][x]/COLOR_HIGH);
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

    scale = atof(argv[1]);
    printf("Scale %f\n", scale);
    acc_time = (float)(atoi(argv[2]))/1000;

    int scnn_id[NB_NETS];
    for (int i = 0; i < NB_NETS; i++) {
        scnn_id[i] = i;
        x_px[i] = WINDOW_WIDTH/2;
        y_px[i] = SLICE_HEIGHT*(i+0.5);
    }

    memset(input_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(shared_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(emptyMatrix, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);

    pthread_t updateRawThread, updateSharedThread, updateCnnThreads[NB_NETS];
    pthread_t renderThread;


    // Create threads
    pthread_create(&updateRawThread, NULL, updateRaw, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_create(&updateCnnThreads[i], NULL, updateCnn, (void *)&scnn_id[i]);
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
