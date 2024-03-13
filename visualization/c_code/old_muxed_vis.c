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
#define K_SZ 33 // maybe 26?
#define NB_NETS 3
#define NB_SLICES (NB_NETS + 1)
#define EV_COUNT_THRESHOLD 1
#define MAX_DELTA 20 // 10 pixels

#define WINDOW_WIDTH 256
#define SLICE_HEIGHT 164
#define WINDOW_HEIGHT SLICE_HEIGHT*NB_SLICES

#define MAT_BUFF_SIZE 1024 * 256
#define PORT_UDP_RAW 3330 // Port through which input events are received (from camera using AEstream)
#define PORT_UDP_CNN 3331 // Port through which output events are received (from SpiNNaker using SPIF)


#define RECEIVER_IP "172.16.222.30"
#define PORT_UDP_TIP_TARGET 6161
#define PORT_UDP_TIP_CURRENT 7373

// Colors
int base = 0xFF000000;
int black = 0x00000000;
int green = 0x00006400;
int blue = 0x000000FF;
int cyan = 0x0000FFFF;
int magenta = 0x00FF66FF;
int red = 0x00FF0000;
int yellow = 0x00FFFF00;

int emptyMatrix[WINDOW_HEIGHT][WINDOW_WIDTH];


float acc_time = 0.001000;



float scale = 1.0; // No need for mutex since it's not accessed by parallel processes


// @TODO: Maybe add mutex for these variables?????
int ev_count[NB_SLICES];
struct timespec last_ev_count[NB_SLICES];


/***********************************************************************************************************/

pthread_mutex_t raw_mutex = PTHREAD_MUTEX_INITIALIZER;
int input_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // where incoming raw data is stored
int shared_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix shared between udp receiver (raw) and render|saver
int visual_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used by render (raw)


pthread_mutex_t cnn_mutex = PTHREAD_MUTEX_INITIALIZER;
int input_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // where incoming scnn data is stored
int shared_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix shared between udp receiver (scnn) and render|saver
int visual_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used by render (scnn)


pthread_mutex_t mux_mutex = PTHREAD_MUTEX_INITIALIZER;
int cur_x = WINDOW_WIDTH/2;
int cur_y = SLICE_HEIGHT/2;
int mem_x = WINDOW_WIDTH/2;
int mem_y = SLICE_HEIGHT*(NB_SLICES-1)+SLICE_HEIGHT/2;



pthread_mutex_t xyp_mutex = PTHREAD_MUTEX_INITIALIZER;
int x_px[NB_SLICES];
int y_px[NB_SLICES];



pthread_mutex_t end_mutex = PTHREAD_MUTEX_INITIALIZER;
bool is_the_end = false;

/***********************************************************************************************************/




void die(char *s) {
    perror(s);
    exit(1);
}

// Function to calculate Euclidean distance between two points
double get_distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

/* This process updates the shared raw matrices with events coming from camera (through UDP)*/
void* updateRaw(void* arg) {
    
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[MAT_BUFF_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    } else {
        printf("Socket for RAW correctly created\n");
    }

    int port_raw = PORT_UDP_RAW;

    // Initialize server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_raw); // Use the same port as in Python

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for CNN data on port %d...\n", port_raw);

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
        
    }

    close(sockfd);

    return NULL;
}


/* This process updates the shared cnn matrices with events coming from SpiNNaker (through UDP)*/
void* updateCnn(void* arg) {
    

    int screen_id = *((int *)arg);
    printf("Screen ID: %d\n", screen_id);

    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[MAT_BUFF_SIZE];

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

    float tip_x, tip_y;
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
            // time(&last_ev_count[screen_id]);
            clock_gettime(CLOCK_MONOTONIC, &last_ev_count[screen_id]);
            ///printf("Time[%d]: %ld\n", screen_id, last_ev_count[screen_id]);
            pthread_mutex_unlock(&xyp_mutex);  
        }        

    }

    // close(sock_tip);
    close(sockfd);

    return NULL;
}

void* monitorEvCount(void* arg){

    struct timespec current_t;
    double t_diff_ms[NB_NETS];

    while(1){
    clock_gettime(CLOCK_MONOTONIC, &current_t);
        pthread_mutex_lock(&xyp_mutex);  
        for(int screen_id = 0; screen_id < NB_NETS; screen_id++){
            t_diff_ms[screen_id] = (current_t.tv_sec - last_ev_count[screen_id].tv_sec) * 1000.0 +
                              (current_t.tv_nsec - last_ev_count[screen_id].tv_nsec) / 1e6;
            
            if(t_diff_ms[screen_id] > 20){
                ev_count[screen_id] = 0;
            }
        }
        // printf("Diff: %ld | %ld | %ld | %ld\n", current_t, last_ev_count[0], last_ev_count[1], last_ev_count[2]);
        // printf("Diff: %f | %f | %f\n", t_diff_ms[0], t_diff_ms[1], t_diff_ms[2]);
        pthread_mutex_unlock(&xyp_mutex);  
    }
}

void* updateShared(void* arg) {

    bool local_end = false;

    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  

        usleep(1000000*acc_time);

        // Acquire mutex lock | update data | release mutex lock
        pthread_mutex_lock(&raw_mutex);
        memcpy(shared_raw_mat, input_raw_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&raw_mutex);

        pthread_mutex_lock(&cnn_mutex);        
        memcpy(shared_cnn_mat, input_cnn_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&cnn_mutex);

        memcpy(input_raw_mat, emptyMatrix, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        memcpy(input_cnn_mat, emptyMatrix, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    }
}


// Function to render the entire matrix as a texture
void* multiplex(void* arg) {



}

// Function to render the entire matrix as a texture
void* renderMatrix(void* arg) {


    int sock;
    struct sockaddr_in receiver_addr;
    
    // Create UDP socket
    if ((sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    
    memset(&receiver_addr, 0, sizeof(receiver_addr));
    receiver_addr.sin_family = AF_INET;
    receiver_addr.sin_port = htons(PORT_UDP_TIP_TARGET);
    
    if (inet_aton(RECEIVER_IP, &receiver_addr.sin_addr) == 0) {
        fprintf(stderr, "Invalid address\n");
        exit(EXIT_FAILURE);
    }


    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Air Hockey", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, scale*WINDOW_WIDTH, scale*SLICE_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* matrixTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, SLICE_HEIGHT);


    bool local_end = false;
    int local_x_px[NB_SLICES];
    int local_y_px[NB_SLICES];
    int local_ev_count[NB_SLICES];
    int mux_id = NB_SLICES-1;

    float fraction_x;
    float fraction_y;
    float old_fraction_x;
    float old_fraction_y;

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

        // printf("Ev Counts: %d | %d | %d \n", local_ev_count[0], local_ev_count[1], local_ev_count[2]);


        bool activity_detected = false;
        mux_id = NB_NETS;
        for(int screen_id = 0; screen_id < NB_NETS; screen_id++){
            if(local_ev_count[screen_id] > EV_COUNT_THRESHOLD){
                mux_id = screen_id;
                activity_detected = true;
                break;
            }
        }



        memcpy(visual_cnn_mat[SLICE_HEIGHT*(NB_SLICES-1)], shared_cnn_mat[SLICE_HEIGHT*mux_id], sizeof(int) * WINDOW_WIDTH * SLICE_HEIGHT);
        local_x_px[NB_SLICES-1] = local_x_px[mux_id];
        local_y_px[NB_SLICES-1] = local_y_px[mux_id]-mux_id*SLICE_HEIGHT+NB_NETS*SLICE_HEIGHT;
               

        cur_x = local_x_px[NB_SLICES-1];
        cur_y = local_y_px[NB_SLICES-1];
        // if(get_distance(cur_x, cur_y, mem_x, mem_y)> MAX_DELTA){
        //     cur_x = mem_x;
        //     cur_y = mem_y;
        // }
        // mem_x = cur_x;
        // mem_y = cur_y;

        fraction_x = (local_x_px[NB_SLICES-1]);
        fraction_x = fraction_x/WINDOW_WIDTH;
        fraction_y = ((local_y_px[NB_SLICES-1]-(NB_SLICES-1)*SLICE_HEIGHT));
        fraction_y = fraction_y/SLICE_HEIGHT;


        if (local_x_px[NB_SLICES-1] == 0 || local_y_px[NB_SLICES-1] == 0){
            fraction_x = old_fraction_x;
            fraction_y = old_fraction_y;
        }

        // printf("Current: x=%f | y=%f\n", fraction_x, fraction_y);

        if(fraction_x > 0 && fraction_y > 0){
            // Encode coordinates as string
            char data[50]; // Sufficient size for float to string conversion
            snprintf(data, sizeof(data), "%.3f,%.3f", fraction_x*100, fraction_y*100);
            // printf("%s\n", data);
            // Send data to receiver
            if (sendto(sock, data, strlen(data), 0, (struct sockaddr *)&receiver_addr, sizeof(receiver_addr)) == -1) {
                perror("sendto");
                exit(EXIT_FAILURE);
            }
        }
        old_fraction_x = fraction_x;
        old_fraction_y = fraction_y;

        

        int mux_color;
        if(mux_id == 0){
            mux_color = cyan;
        } else if (mux_id == 1) {
            mux_color = magenta;
        } else if (mux_id == 2) {
            mux_color = yellow;
        } else {
            mux_color = black;
        }

        // Lock the texture for writing
        void* pixels;
        int pitch;
        SDL_LockTexture(matrixTexture, NULL, &pixels, &pitch);

        int color = black;

        // Update the texture pixel data
        int screen_id = NB_SLICES-1;
        for (int y = SLICE_HEIGHT*(screen_id+0); y < SLICE_HEIGHT*(screen_id+1); y++) {
            for (int x = 0; x < WINDOW_WIDTH; x++) {
                int index = (y-screen_id*SLICE_HEIGHT) * WINDOW_WIDTH + x;
                color = black;
                if (visual_raw_mat[y][x] >= 1) {
                    // Set color to green (0xFF00FF00) if the matrix value is 1
                    color = green*(visual_raw_mat[y][x]/COLOR_HIGH);
                } 
                // if (get_distance(x, y, local_x_px[screen_id], local_y_px[screen_id])<=3){
                //     if(screen_id == 0){
                //         color = cyan;
                //     } else if (screen_id == 1) {
                //         color = magenta;
                //     } else if (screen_id == 2){
                //         color = yellow;
                //     } else {
                //         color = mux_color;
                //     }
                // }
                // if (get_distance(x, y, cur_x, cur_y)<=3){
                //     color = red;
                // }
                if (visual_cnn_mat[y][x] >= 1) {
                    // Set color to green (0xFF00FF00) if the matrix value is 1 
                    color = red*(visual_cnn_mat[y][x]/COLOR_HIGH);
                } 
                ((Uint32*)pixels)[index] = color + base;
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

    pthread_t updateRawThread, updateSharedThread, multiplexerThread, updateCnnThreads[NB_NETS];
    pthread_t renderThread, timerThread;


    // Create threads
    pthread_create(&updateRawThread, NULL, updateRaw, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_create(&updateCnnThreads[i], NULL, updateCnn, (void *)&scnn_id[i]);
    }

    pthread_create(&multiplexerThread, NULL, multiplex, NULL);
    pthread_create(&updateSharedThread, NULL, updateShared, NULL);
    pthread_create(&renderThread, NULL, renderMatrix, NULL);
    pthread_create(&timerThread, NULL, monitorEvCount, NULL);

    // Wait for threads to finish (this will never happen in this example)
    pthread_join(updateRawThread, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_join(updateCnnThreads[i], NULL);
    }
    pthread_join(multiplexerThread, NULL);
    pthread_join(updateSharedThread, NULL);
    pthread_join(renderThread, NULL);
    pthread_join(timerThread, NULL);

    // Clean up mutex
    pthread_mutex_destroy(&raw_mutex);
    pthread_mutex_destroy(&cnn_mutex);

    return 0;
}
