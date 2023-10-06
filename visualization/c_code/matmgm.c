
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <time.h>

#define NB_COLS 16
#define NB_ROWS 16

#define BUFFER_SIZE 1024*256
#define PORT_UDP 9524


int rcvngMatrix[NB_COLS][NB_ROWS];
int readyMatrix[NB_COLS][NB_ROWS];
int emptyMatrix[NB_COLS][NB_ROWS];

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void printMatrix(const int matrix[]) {
    for (int i = 0; i < NB_COLS; i++) {
        for (int j = 0; j < NB_ROWS; j++) {
            printf("%d ", matrix[i * NB_ROWS + j]);
        }
        printf("\n");
    }
}

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

                rcvngMatrix[x][y] = 1;
            }
            
            // Get the current time
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            
            // Calculate the elapsed time in seconds
            elapsed_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            
        } while (elapsed_time < 0.001); // Repeat until 1ms has elapsed
        
        // printf("%d receptions in 1ms\n", recv_counter);

        // Acquire the mutex lock
        pthread_mutex_lock(&mutex);        
        memcpy(readyMatrix, rcvngMatrix, sizeof(int) * NB_COLS * NB_ROWS);
        memcpy(rcvngMatrix, emptyMatrix, sizeof(int) * NB_COLS * NB_ROWS);
        pthread_mutex_unlock(&mutex);

        sleep(2); // Sleep for 2 seconds
    }

    close(sockfd);

    return NULL;
}

// Function to print the shared array
void* printArray(void* arg) {
    while (1) {
        // Acquire the mutex lock
        pthread_mutex_lock(&mutex);

        // Print the shared array
        printMatrix((int *)readyMatrix);
        printf("\n");

        // Release the mutex lock
        pthread_mutex_unlock(&mutex);

        sleep(1); // Sleep for 1 second
    }
    return NULL;
}

int main(int argc, char *argv[]) {


    memset(rcvngMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);
    memset(readyMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);
    memset(emptyMatrix, 0, sizeof(int) * NB_COLS * NB_ROWS);







    pthread_t updateThread, printThread;

    // Create threads
    pthread_create(&updateThread, NULL, updateArray, NULL);
    pthread_create(&printThread, NULL, printArray, NULL);

    // Wait for threads to finish (this will never happen in this example)
    pthread_join(updateThread, NULL);
    pthread_join(printThread, NULL);

    // Clean up mutex
    pthread_mutex_destroy(&mutex);

    return 0;
}
