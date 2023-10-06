#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <time.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define WINDOW_WIDTH 280
#define WINDOW_HEIGHT 181

#define NB_COLS 280
#define NB_ROWS 181
#define NB_FRAMES 5000

#define BUFFER_SIZE 1024 
#define PORT_UDP 9524

int frame_counter = 0;
int inputMatrix[NB_COLS][NB_ROWS];
int outputMatrix[NB_COLS][NB_ROWS];
int sharedMatrix[NB_COLS][NB_ROWS];
int emptyMatrix[NB_COLS][NB_ROWS];
int videoMatrix[NB_FRAMES][NB_COLS][NB_ROWS];

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* updateArray(void* arg) {
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
    server_addr.sin_port = htons(PORT_UDP); // Use the same port as in Python

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for data...\n");

    f  d_set readfds;
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 1000; // 1000 microseconds (1 millisecond) timeout

    int ready = select(sockfd + 1, &readfds, NULL, NULL, &timeout);

    while (1) {
    
        
        int recv_counter = 0;
        clock_t loop_start, loop_end;    
        loop_start = clock();

        while(recv_counter<=240){
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

                inputMatrix[x][y] = 1;
            }            
        } 
                      
        loop_end = clock();        
        double loop_time = (double)(loop_end - loop_start) / CLOCKS_PER_SEC;
        printf("Loop time: %lf[s]\n", loop_time);

        // Acquire mutex lock | update data | release mutex lock
        pthread_mutex_lock(&mutex);        
        memcpy(sharedMatrix, inputMatrix, sizeof(int) * NB_COLS * NB_ROWS);
        pthread_mutex_unlock(&mutex);
        frame_counter++;

        memcpy(inputMatrix, emptyMatrix, sizeof(int) * NB_COLS * NB_ROWS);

    }

    close(sockfd);

    return NULL;
}


void* storeMatrix(void* arg) {
    
    while(1){
        // pthread_mutex_lock(&mutex);
        // if(frame_counter >= NB_FRAMES){
        //     break;
        // }
        // memcpy(outputMatrix, sharedMatrix, sizeof(int) * NB_COLS * NB_ROWS);        
        // pthread_mutex_unlock(&mutex);      
        // // printf("frame counter : %d\n", frame_counter);  
        // memcpy(videoMatrix[frame_counter], outputMatrix, sizeof(int) * NB_COLS * NB_ROWS);   
        int a= 1;
    }
    printf("Done Storing ");

}

int main(int argc, char* argv[]) {
    memset(inputMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);
    memset(sharedMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);
    memset(emptyMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);

    pthread_t updateThread, renderThread;

    // Create threads
    pthread_create(&updateThread, NULL, updateArray, NULL);
    pthread_create(&renderThread, NULL, storeMatrix, NULL);

    // Wait for threads to finish (this will never happen in this example)
    pthread_join(updateThread, NULL);
    pthread_join(renderThread, NULL);

    // Clean up mutex
    pthread_mutex_destroy(&mutex);

    return 0;
}
