#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>


#define TIP_BUFF_SIZE 1024  // Max length of buffer
#define TIP_PORT 7373   // The TIP_PORT on which to listen for incoming data

void die(char *s) {
    perror(s);
    exit(1);
}

int main(void) {

    float x, y;

    struct sockaddr_in si_me, si_other;
    int sock_tip, i, slen = sizeof(si_other);
    char buf[TIP_BUFF_SIZE];

    // Create a UDP socket
    if ((sock_tip=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        die("socket");
    }

    // Zero out the structure
    memset((char *) &si_me, 0, sizeof(si_me));

    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(TIP_PORT);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);

    // Bind socket to TIP_PORT
    if (bind(sock_tip, (struct sockaddr*)&si_me, sizeof(si_me)) == -1) {
        die("bind");
    }

    // Receive data loop
    while (1) {
        // Clear the buffer
        memset(buf, 0, TIP_BUFF_SIZE);

        // Try to receive some data
        if (recvfrom(sock_tip, buf, TIP_BUFF_SIZE, 0, (struct sockaddr*)&si_other, &slen) == -1) {
            die("recvfrom()");
        }

        // Print received data
        memcpy(&x, &buf[0], sizeof(float));
        memcpy(&y, &buf[sizeof(float)], sizeof(float));

        // Print received data
        printf("Received packet from %s:%d\n", inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port));
        printf("Data: x = %f, y = %f\n" , x, y);
    }

    close(sock_tip);
    return 0;
}
