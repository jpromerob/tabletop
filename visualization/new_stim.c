#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <arpa/inet.h>

#define IP "172.16.222.199"
#define PORT 3330
#define NUM_EVENTS 1 // Number of events to send

void generateEvents(int cx, int cy, int r) {
    int numPoints = 360;
    int pShift = 15;
    int yShift = 0;
    int xShift = 16;
    unsigned int noTimestamp = 0x80000000;
    int sock;
    struct sockaddr_in server;

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
    server.sin_addr.s_addr = inet_addr(IP);

    int polarity = 1;
    int step = 0;

    unsigned int *eventArray = (unsigned int *)malloc(NUM_EVENTS * sizeof(unsigned int));
    if (eventArray == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    while(1){
        for (int i = 0; i < NUM_EVENTS; i++) {
            double angle = 2 * M_PI * step / numPoints;
            int x = cx + r * cos(angle);
            int y = cy + r * sin(angle);
            step = (step + 1) % numPoints;

            eventArray[i] = noTimestamp + (polarity << pShift) + (y << yShift) + (x << xShift);
            // printf("Sending event: Polarity=%u, X=%d, Y=%d\n", polarity, x, y);
            usleep(200); // Sleep for 0.2 milliseconds (200 microseconds)
        }

        // Send the entire array
        sendto(sock, eventArray, NUM_EVENTS * sizeof(unsigned int), 0, (struct sockaddr *)&server, sizeof(server));

    }

    free(eventArray);
    close(sock);
}

int main() {
    int cx = 200;
    int cy = 100;
    int r = 10;

    generateEvents(cx, cy, r);

    return 0;
}
