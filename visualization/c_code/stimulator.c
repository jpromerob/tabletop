#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <arpa/inet.h>

#define IP "172.16.222.199"
#define PORT 3331

void generateEvent(int cx, int cy, int r) {
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
    
    while (1) {
        double angle = 2 * M_PI * step / numPoints;
        int x = cx + r * cos(angle);
        int y = cy + r * sin(angle);
        step = (step + 1) % numPoints;
        
        unsigned int packed = noTimestamp + (polarity << pShift) + (y << yShift) + (x << xShift);
        sendto(sock, &packed, sizeof(packed), 0, (struct sockaddr*)&server, sizeof(server));
        
        usleep(200);  // Sleep for 0.2 milliseconds (200 microseconds)
    }

    close(sock);
}

int main() {
    int cx = 200;
    int cy = 100;
    int r = 10;

    generateEvent(cx, cy, r);

    return 0;
}