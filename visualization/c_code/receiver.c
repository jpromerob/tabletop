#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <pthread.h>
// #include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PORT 3330


#define ROWS 480
#define COLS 640
int array[ROWS][COLS];
int empty[ROWS][COLS];
int ready[ROWS][COLS];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Function for the "hola" thread
void *p_reception(void *arg) {
    


    struct timespec start, end;
    long long elapsed_time_ns;

    double time_th = 2.0;

    // Initialize all elements of the array to 0
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            empty[i][j] = 0;
        }
    }

    // Use memcpy to copy the contents of 'empty' to 'array'
    memcpy(array, empty, sizeof(array));


    int sockfd;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t addrLen = sizeof(clientAddr);
    unsigned int buffer;

    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Configure server address
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) == -1) {
        perror("Binding failed");
        exit(EXIT_FAILURE);
    }

    printf("Listening for events on port %d...\n", PORT);

    clock_gettime(CLOCK_MONOTONIC, &start);
    while (1) {
        // Receive data into the buffer
        ssize_t bytesRead = recvfrom(sockfd, &buffer, sizeof(buffer), 0, (struct sockaddr *)&clientAddr, &addrLen);
        if (bytesRead == -1) {
            perror("Error receiving data");
            continue;
        }
        
        unsigned int polarityMask = 0x40000000;
        unsigned int xMask = 0xFFFC0000;
        unsigned int yMask = 0x00003FFF;

        unsigned int polarity = (buffer & polarityMask) >> 30;
        int x = (buffer >> 16) & 0x00003FFF;
        int y = buffer & yMask;

        array[y][x] += 1;

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed_time_ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
        double elapsed_time_ms = (double)elapsed_time_ns / 1000000.0;

        // printf("Elapsed Time: %lf seconds\n", elapsed_time_ms);
    
        if (elapsed_time_ms >= time_th*2000) {
            start = end; // Reset the start time
            pthread_mutex_lock(&mutex);
            memcpy(ready, array, sizeof(array));
            pthread_mutex_unlock(&mutex);
            memcpy(array, empty, sizeof(array));
            // printf("Clear matrix\n");
        }
        printf("Received event: Polarity=%u, X=%d, Y=%d\n", polarity, x, y);
    }

    close(sockfd);

    return NULL;
}

int main() {
    pthread_t p_reception_id;
    
    // Create and start the "hola" thread
    if (pthread_create(&p_reception_id, NULL, p_reception, NULL) != 0) {
        perror("pthread_create");
        return 1;
    }

    // Main thread: Print "salut" every 2 seconds
    while (1) {
        pthread_mutex_lock(&mutex);
        printf("salut\n");
        pthread_mutex_unlock(&mutex);
        sleep(2);
    }

    return 0;
}