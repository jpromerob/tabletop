
#include <stdio.h>
#include <stdlib.h>
#include <SDL.h>
#include <string.h>
#include <arpa/inet.h>
#include <time.h>

// Define the dimensions of the window and the matrix
#define WINDOW_WIDTH 252
#define WINDOW_HEIGHT 153
#define MATRIX_ROWS 153
#define MATRIX_COLS 252

#define BUFFER_SIZE 1024*32


// Function to create a 2D matrix
int **createMatrix(int rows, int cols) {
    int **matrix;

    matrix = (int **)malloc(rows * sizeof(int *));
    if (matrix == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = (int *)malloc(cols * sizeof(int));
        if (matrix[i] == NULL) {
            perror("Memory allocation failed");
            exit(1);
        }
    }

    return matrix;
}

// Function to initialize the matrix with values
void initializeMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0;
        }
    }
}

// Function to print the matrix
void printMatrix(int **matrix, int rows, int cols) {
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to deallocate memory used by the matrix
void freeMatrix(int **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}



int main() {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    // Create a window
    SDL_Window *window = SDL_CreateWindow("Matrix Visualization",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          WINDOW_WIDTH,
                                          WINDOW_HEIGHT,
                                          0);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Create a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Create and initialize the matrix
    


    int rows = WINDOW_WIDTH;
    int cols = WINDOW_HEIGHT;

    int **matrix = createMatrix(cols, rows);
    int **empty = createMatrix(cols, rows);
    initializeMatrix(matrix, cols, rows);
    initializeMatrix(empty, cols, rows);

    // copyMatrix(matrix, empty, MATRIX_ROWS, MATRIX_COLS);

    SDL_Event event;
    int quit = 0;


    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    unsigned char buffer[BUFFER_SIZE];

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    }

    // Initialize server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(9524); // Use the same port as in Python

    // Bind socket to server address
    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(1);
    }

    printf("Listening for data...\n");

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = 1;
            }
        }

        clock_t start_time = clock();        
        int ms_counter = 0;
        while (1) {
            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;
            
            if (elapsed_time >= 0.001) {

                ssize_t recv_len = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                                        (struct sockaddr *)&client_addr, &client_addr_len);

                if (recv_len < 0) {
                    perror("recvfrom");
                    exit(1);
                }

                // Assuming the received data is packed as little-endian 32-bit integers
                for (int i = 0; i < recv_len; i += 4) {
                    unsigned int packed_data;
                    memcpy(&packed_data, &buffer[i], 4);


                    // Extract x and y values
                    int x = (packed_data >> 16) & 0x00003FFF;
                    int y = (packed_data >> 0) & 0x00003FFF;

                    if((x>=0 && x < WINDOW_WIDTH) && (y>=0 && y< WINDOW_HEIGHT)){
                        matrix[x][y] = 128;
                    }
                }
            } else {
                if (ms_counter+1 < 20) {
                    ms_counter +=1;

                } else {
                    ms_counter= 0;
                    break;
                }
            }
            
        }
        

        // printf("Visualizing\n");

        // Clear the renderer
        SDL_RenderClear(renderer);

        // Render the matrix as a grid of rectangles
        for (int i = 0; i < MATRIX_ROWS; i++) {
            for (int j = 0; j < MATRIX_COLS; j++) {
                SDL_Rect rect = {j * (WINDOW_WIDTH / MATRIX_COLS),
                                 i * (WINDOW_HEIGHT / MATRIX_ROWS),
                                 WINDOW_WIDTH / MATRIX_COLS,
                                 WINDOW_HEIGHT / MATRIX_ROWS};
                SDL_SetRenderDrawColor(renderer,
                                       matrix[i][j] % 256,  // R
                                       matrix[i][j] % 256,  // G
                                       matrix[i][j] % 256,   // B
                                       255);  // Alpha
                SDL_RenderFillRect(renderer, &rect);
            }
        }

        // Update the screen
        SDL_RenderPresent(renderer);
    }

    freeMatrix(matrix, cols);

    // Cleanup and quit SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    close(sockfd);

    return 0;
}
