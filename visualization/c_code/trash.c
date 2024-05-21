#include <stdio.h>

void incrementAndPrint() {
    static int staticVariable = 0; // Static local variable initialized to 0
    
    staticVariable++; // Increment the static variable
    printf("Static variable value: %d\n", staticVariable);
}

int main() {
    incrementAndPrint(); // Call the function
    incrementAndPrint(); // Call the function again
    incrementAndPrint(); // Call the function once more
    
    return 0;
}