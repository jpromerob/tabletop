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
#define SLICE_HEIGHT 165
#define WINDOW_HEIGHT SLICE_HEIGHT*NB_SLICES

#define MAT_BUFF_SIZE 1024 * 256
#define TIPPOS_BUFF_SIZE 512

#define PORT_UDP_RAW 3330 // Port through which input events are received (from camera using AEstream)
#define PORT_UDP_CNN 3331 // Port through which output events are received (from SpiNNaker using SPIF)


#define RECEIVER_IP "172.16.222.30"
#define PORT_UDP_TIP_TARGET 6161
#define PORT_UDP_TIP_CURRENT 6363
#define PORT_UDP_TIP_DESIRED 6464
#define PADDLE_RADIUS 14

// Colors
#define BASE_COLOR 0xFF000000
#define BLACK 0x00000000
#define GREEN 0x00006400
#define BLUE 0x000000FF
#define CYAN 0x0000FFFF
#define MAGENTA 0x00FF66FF
#define RED 0x00FF0000
#define YELLOW 0x00FFFF00
#define WHITE 0x00FFFFFF
#define ORANGE 0x00FFA500


int emptyMatrix[WINDOW_HEIGHT][WINDOW_WIDTH];
float acc_time_in_s = 0.001000; /* 1[ms]*/
float scale = 1.0; // No need for mutex since it's not accessed by parallel processes


/**********************************************************************************************************

This block defines mutexes and variables that require their use (given parallel access)

***********************************************************************************************************/

pthread_mutex_t raw_mutex = PTHREAD_MUTEX_INITIALIZER;
int input_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // where incoming raw data is stored
int shared_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix shared between udp receiver (raw) and render|saver
int visual_raw_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used by render (raw)


pthread_mutex_t cnn_mutex = PTHREAD_MUTEX_INITIALIZER;
int input_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // where incoming scnn data is stored
int shared_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix shared between udp receiver (scnn) and render|saver
int visual_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used by render (scnn)
int location_cnn_mat[WINDOW_HEIGHT][WINDOW_WIDTH]; // matrix used to find XY location

pthread_mutex_t mux_mutex = PTHREAD_MUTEX_INITIALIZER;
int cur_x = WINDOW_WIDTH/2;
int cur_y = SLICE_HEIGHT/2;
int mux_color = WHITE;
int mux_id = NB_NETS;


pthread_mutex_t cur_tip_mutex = PTHREAD_MUTEX_INITIALIZER;
int cur_tip_x = WINDOW_WIDTH/2;
int cur_tip_y = SLICE_HEIGHT/2;

pthread_mutex_t des_tip_mutex = PTHREAD_MUTEX_INITIALIZER;
int des_tip_x = WINDOW_WIDTH/2;
int des_tip_y = SLICE_HEIGHT/2;

pthread_mutex_t xyp_mutex = PTHREAD_MUTEX_INITIALIZER;
int x_px[NB_SLICES];
int y_px[NB_SLICES];


// @TODO: Maybe add mutex for these variables?????
pthread_mutex_t ev_count_mutex = PTHREAD_MUTEX_INITIALIZER;
int ev_count[NB_SLICES];
struct timespec last_ev_count[NB_SLICES];

pthread_mutex_t end_mutex = PTHREAD_MUTEX_INITIALIZER;
bool is_the_end = false;

/***********************************************************************************************************/




void die(char *s) {
    perror(s);
    exit(1);
}

float from_s_to_us(float value_in_s){
    float value_in_us;
    value_in_us = 1000000*value_in_s;
    return value_in_us;
}


// Function to calculate Euclidean distance between two points
double get_distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}




/****************************************************************************

This process updates the shared raw matrices with events coming from 
camera (through UDP)

*****************************************************************************/
void* updateRaw(void* arg) {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[MAT_BUFF_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    }

    int port_raw = PORT_UDP_RAW;

    // Initialize server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_raw);

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for RAW data on port %d\n", port_raw);

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
            
        } while (elapsed_time < acc_time_in_s); // Repeat until 1ms has elapsed
        
    }

    close(sockfd);

    return NULL;
}


/****************************************************************************

This process updates the shared cnn matrices with events coming from 
SpiNNaker (through UDP)

*****************************************************************************/

void* updateCnn(void* arg) {
    
    usleep(100*1000);

    int screen_id = *((int *)arg);

    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[MAT_BUFF_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
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

    float cur_tip_x, cur_tip_y;
    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  
        double elapsed_time;
    
        // Get the current time
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    
        
        int recv_counter = 0;

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
                int x = ((packed_data >> 16) & 0x00003FFF);
                int y = ((packed_data >> 0) & 0x00003FFF);

                input_cnn_mat[y+SLICE_HEIGHT*screen_id][x] += 1;
            }
            
            // Get the current time
            clock_gettime(CLOCK_MONOTONIC, &current_time);

            // Calculate the elapsed time in seconds
            elapsed_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
        } while (elapsed_time < acc_time_in_s); // Repeat until 1ms has elapsed

    }

    // close(sock_tip);
    close(sockfd);

    return NULL;
}

/****************************************************************************

This process sets event count to zero every once in a while

*****************************************************************************/
void* monitorEvCount(void* arg){

    struct timespec current_t;
    int refresh_time_ms = 20;
    double t_diff_ms[NB_NETS];


    while(1){
    clock_gettime(CLOCK_MONOTONIC, &current_t);
        pthread_mutex_lock(&ev_count_mutex);  
        for(int screen_id = 0; screen_id < NB_NETS; screen_id++){
            t_diff_ms[screen_id] = (current_t.tv_sec - last_ev_count[screen_id].tv_sec) * 1000.0 +
                              (current_t.tv_nsec - last_ev_count[screen_id].tv_nsec) / 1e6;
            
            if(t_diff_ms[screen_id] > refresh_time_ms){
                ev_count[screen_id] = 0;
            }
        }
        pthread_mutex_unlock(&ev_count_mutex);  
    }
}


/****************************************************************************

This process updates shared matrices and clears input matrices

*****************************************************************************/
void* updateShared(void* arg) {

    bool local_end = false;

    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  
        
        pthread_mutex_lock(&raw_mutex);
        memcpy(shared_raw_mat, input_raw_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&raw_mutex);

        pthread_mutex_lock(&cnn_mutex);        
        memcpy(shared_cnn_mat, input_cnn_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        memcpy(location_cnn_mat, input_cnn_mat, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        pthread_mutex_unlock(&cnn_mutex);


        memcpy(input_raw_mat, emptyMatrix, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
        memcpy(input_cnn_mat, emptyMatrix, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);

        usleep(from_s_to_us(acc_time_in_s));
    }
}

/****************************************************************************

This process in in charge of finding the hotspots (of activity) for each of
the SCNNs' outputs. The idea is to accumulate events (count) for some time
and then return average indices (for x and y) of activity.

*****************************************************************************/
void* findLocation(void* arg) {


    int screen_id = *((int *)arg);
    printf("Finding Location for Screen Id %d\n", screen_id);

    usleep((1000+screen_id+1)*1000);

    bool local_end = false;
    int local_cur_tip_x;
    int local_cur_tip_y;

    while (!local_end) {

        int sum_idx_x = 0;
        int sum_idx_y = 0;
        int count_ones = 0;

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);  


        for (int y = SLICE_HEIGHT*(screen_id+0); y < SLICE_HEIGHT*(screen_id+1); y++) {
            for (int x = 0; x < WINDOW_WIDTH; x++) {
                if(location_cnn_mat[y][x]>0){                    
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

            pthread_mutex_lock(&ev_count_mutex); 
            ev_count[screen_id] = count_ones;
            clock_gettime(CLOCK_MONOTONIC, &last_ev_count[screen_id]);
            pthread_mutex_unlock(&ev_count_mutex);  
        }   

        usleep(from_s_to_us(acc_time_in_s));

    }

}

/****************************************************************************

This process is in charge of 'multiplexing' SCNN outputs.
The idea is to consolidate a one and only (x,y) coordinate

*****************************************************************************/
void* multiplex(void* arg) {

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

    int local_x_px[NB_SLICES];
    int local_y_px[NB_SLICES];
    int local_ev_count[NB_SLICES];

    float fraction_x;
    float fraction_y;
    float old_fraction_x;
    float old_fraction_y;

    bool local_end = false;

    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);


        pthread_mutex_lock(&xyp_mutex);          
        memcpy(local_x_px, x_px, sizeof(int) * NB_SLICES);
        memcpy(local_y_px, y_px, sizeof(int) * NB_SLICES);
        memcpy(local_ev_count, ev_count, sizeof(int) * NB_SLICES);
        pthread_mutex_unlock(&xyp_mutex);   

        bool activity_detected = false;

               

        pthread_mutex_lock(&mux_mutex);    
        
        for(int screen_id = 0; screen_id < NB_NETS; screen_id++){
            if(local_ev_count[screen_id] > EV_COUNT_THRESHOLD){
                mux_id = screen_id;
                activity_detected = true;
                break;
            }
        }

        memcpy(visual_cnn_mat[SLICE_HEIGHT*(NB_SLICES-1)], shared_cnn_mat[SLICE_HEIGHT*mux_id], sizeof(int) * WINDOW_WIDTH * SLICE_HEIGHT);

        /* Current (x, y) only changes when there is some activity in at elast one of the SCNNs*/
        if(mux_id < NB_NETS){
            local_x_px[NB_SLICES-1] = local_x_px[mux_id];
            local_y_px[NB_SLICES-1] = local_y_px[mux_id]-mux_id*SLICE_HEIGHT+NB_NETS*SLICE_HEIGHT;
        }

        /* Update Color is necessary (i.e. if any SCNN reports activity)*/
        if(mux_id == 0){
            mux_color = CYAN;
        } else if (mux_id == 1) {
            mux_color = MAGENTA;
        } else if (mux_id == 2) {
            mux_color = YELLOW;
        }

        cur_x = local_x_px[NB_SLICES-1];
        cur_y = local_y_px[NB_SLICES-1];
        
        pthread_mutex_unlock(&mux_mutex);

        fraction_x = (local_x_px[NB_SLICES-1]);
        fraction_x = fraction_x/WINDOW_WIDTH;
        fraction_y = ((local_y_px[NB_SLICES-1]-(NB_SLICES-1)*SLICE_HEIGHT));
        fraction_y = fraction_y/SLICE_HEIGHT;


        if (local_x_px[NB_SLICES-1] == 0 || local_y_px[NB_SLICES-1] == 0){
            fraction_x = old_fraction_x;
            fraction_y = old_fraction_y;
        }


        if(fraction_x > 0 && fraction_y > 0){

            // Encode coordinates as string
            char data[50]; // Sufficient size for float to string conversion
            snprintf(data, sizeof(data), "%.3f,%.3f", fraction_x*100, fraction_y*100);
            
            // Send data to receiver
            if (sendto(sock, data, strlen(data), 0, (struct sockaddr *)&receiver_addr, sizeof(receiver_addr)) == -1) {
                perror("sendto");
                exit(EXIT_FAILURE);
            }
        }
        old_fraction_x = fraction_x;
        old_fraction_y = fraction_y;

        


    }

}



/****************************************************************************

*****************************************************************************/
void* renderMatrix(void* arg) {


    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Air Hockey", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, scale*WINDOW_WIDTH, scale*SLICE_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* matrixTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WINDOW_WIDTH, SLICE_HEIGHT);


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



        int local_cur_x;
        int local_cur_y;
        int local_mux_color;
        pthread_mutex_lock(&mux_mutex);          
        local_cur_x = cur_x;       
        local_cur_y = cur_y;       
        local_mux_color = mux_color;
        pthread_mutex_unlock(&mux_mutex);

        int local_cur_tip_x;
        int local_cur_tip_y;
        pthread_mutex_lock(&cur_tip_mutex);  
        local_cur_tip_x = cur_tip_x;
        local_cur_tip_y = cur_tip_y;
        pthread_mutex_unlock(&cur_tip_mutex);


        int local_des_tip_x;
        int local_des_tip_y;
        pthread_mutex_lock(&des_tip_mutex);  
        local_des_tip_x = des_tip_x;
        local_des_tip_y = des_tip_y;
        pthread_mutex_unlock(&des_tip_mutex);
        // printf("Rendering : %d, %d\n", local_cur_tip_x, local_cur_tip_y);

        // Lock the texture for writing
        void* pixels;
        int pitch;
        SDL_LockTexture(matrixTexture, NULL, &pixels, &pitch);

        int color = BLACK;

        // Update the texture pixel data
        int screen_id = NB_SLICES-1;
        for (int y = SLICE_HEIGHT*(screen_id+0); y < SLICE_HEIGHT*(screen_id+1); y++) {
            for (int x = 0; x < WINDOW_WIDTH; x++) {
                int index = (y-screen_id*SLICE_HEIGHT) * WINDOW_WIDTH + x;
                color = BLACK;
                if (visual_raw_mat[y][x] >= 1) {
                    // Set color to GREEN (0xFF00FF00) if the matrix value is 1
                    color = GREEN*(visual_raw_mat[y][x]/COLOR_HIGH);
                } 
                double d_puck = get_distance(x, y, local_cur_x, local_cur_y);
                if (d_puck >K_SZ*4/10-1 && d_puck < K_SZ*4/10+1){
                    color = local_mux_color;
                }
                double d_cur_tip = get_distance(x, y-SLICE_HEIGHT*screen_id, local_cur_tip_x, local_cur_tip_y);
                if (d_cur_tip > PADDLE_RADIUS-1 && d_cur_tip < PADDLE_RADIUS+1){
                    color = BLUE;
                }
                double d_des_tip = get_distance(x, y-SLICE_HEIGHT*screen_id, local_des_tip_x, local_des_tip_y);
                if (d_des_tip < 4){
                    color = ORANGE;
                }
                if (visual_cnn_mat[y][x] >= 1) {
                    color = RED*(visual_cnn_mat[y][x]/COLOR_HIGH);
                } 
                ((Uint32*)pixels)[index] = color + BASE_COLOR;
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

void* updateCurrentTipPosition(void* arg) {
 struct sockaddr_in si_me, si_other;
    int s, i, slen = sizeof(si_other);
    char buf[TIPPOS_BUFF_SIZE];

    if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("socket");
        exit(1);
    }

    memset((char *)&si_me, 0, sizeof(si_me));
    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(PORT_UDP_TIP_CURRENT);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(s, (struct sockaddr *)&si_me, sizeof(si_me)) == -1) {
        perror("bind");
        exit(1);
    }

    bool local_end = false;

    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);
        
        if (recvfrom(s, buf, TIPPOS_BUFF_SIZE, 0, (struct sockaddr *)&si_other, &slen) == -1) {
            perror("recvfrom");
            exit(1);
        }

        // Decode message into two floats
        float x, y;
        sscanf(buf, "%f,%f", &x, &y);

        pthread_mutex_lock(&cur_tip_mutex);  
        cur_tip_x = (int)(x*WINDOW_WIDTH/100);
        cur_tip_y = (int)(y*SLICE_HEIGHT/100);
        pthread_mutex_unlock(&cur_tip_mutex);

    }

    close(s);
}

void* updateDesiredTipPosition(void* arg) {
 struct sockaddr_in si_me, si_other;
    int s, i, slen = sizeof(si_other);
    char buf[TIPPOS_BUFF_SIZE];

    if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("socket");
        exit(1);
    }

    memset((char *)&si_me, 0, sizeof(si_me));
    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(PORT_UDP_TIP_DESIRED);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(s, (struct sockaddr *)&si_me, sizeof(si_me)) == -1) {
        perror("bind");
        exit(1);
    }

    bool local_end = false;

    while (!local_end) {

        pthread_mutex_lock(&end_mutex);  
        local_end = is_the_end;
        pthread_mutex_unlock(&end_mutex);
        
        if (recvfrom(s, buf, TIPPOS_BUFF_SIZE, 0, (struct sockaddr *)&si_other, &slen) == -1) {
            perror("recvfrom");
            exit(1);
        }

        // Decode message into two floats
        float x, y;
        sscanf(buf, "%f,%f", &x, &y);

        pthread_mutex_lock(&des_tip_mutex);  
        des_tip_x = (int)(x*WINDOW_WIDTH/100);
        des_tip_y = (int)(y*SLICE_HEIGHT/100);
        pthread_mutex_unlock(&des_tip_mutex);

    }

    close(s);
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <scale> <acc_time_in_s_ms>\n", argv[0]);
        return 1;
    }

    scale = atof(argv[1]);
    printf("Scale %f\n", scale);
    acc_time_in_s = (float)(atoi(argv[2]))/1000;

    int scnn_id[NB_NETS];
    for (int i = 0; i < NB_NETS; i++) {
        scnn_id[i] = i;
        x_px[i] = WINDOW_WIDTH/2;
        y_px[i] = SLICE_HEIGHT*(i+0.5);
    }

    memset(input_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(shared_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(visual_raw_mat, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);
    memset(emptyMatrix, 0, sizeof(int) * WINDOW_WIDTH * WINDOW_HEIGHT);

    pthread_t updateRawThread;
    pthread_t updateCnnThreads[NB_NETS];
    pthread_t findLocationThreads[NB_NETS];
    pthread_t updateSharedThread;
    pthread_t updateCurrentTipPosThread;
    pthread_t updateDesiredTipPosThread;
    pthread_t multiplexerThread;
    pthread_t renderThread, timerThread;


    // Create threads
    pthread_create(&updateRawThread, NULL, updateRaw, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_create(&updateCnnThreads[i], NULL, updateCnn, (void *)&scnn_id[i]);
        pthread_create(&findLocationThreads[i], NULL, findLocation, (void *)&scnn_id[i]);
    }

    pthread_create(&multiplexerThread, NULL, multiplex, NULL);
    pthread_create(&updateCurrentTipPosThread, NULL, updateCurrentTipPosition, NULL);
    pthread_create(&updateDesiredTipPosThread, NULL, updateDesiredTipPosition, NULL);
    pthread_create(&updateSharedThread, NULL, updateShared, NULL);
    pthread_create(&renderThread, NULL, renderMatrix, NULL);
    pthread_create(&timerThread, NULL, monitorEvCount, NULL);

    // Wait for threads to finish (this will never happen in this example)
    pthread_join(updateRawThread, NULL);
    for (int i = 0; i < NB_NETS; i++) {
        pthread_join(updateCnnThreads[i], NULL);
    }
    pthread_join(multiplexerThread, NULL);
    pthread_join(updateCurrentTipPosThread, NULL);
    pthread_join(updateDesiredTipPosThread, NULL);
    pthread_join(updateSharedThread, NULL);
    pthread_join(renderThread, NULL);
    pthread_join(timerThread, NULL);

    // Clean up mutex
    pthread_mutex_destroy(&raw_mutex);
    pthread_mutex_destroy(&cnn_mutex);

    return 0;
}
