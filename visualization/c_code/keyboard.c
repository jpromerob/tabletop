#include <ncurses.h>

int main() {
    initscr();           // Initialize ncurses
    keypad(stdscr, TRUE); // Enable arrow key support

    int ch;
    while (1) {
        ch = getch(); // Get a character from the keyboard

        // Check which arrow key was pressed and print a corresponding number
        switch (ch) {
            case KEY_UP:
                printw("Up Arrow Pressed\n");
                break;
            case KEY_DOWN:
                printw("Down Arrow Pressed\n");
                break;
            case KEY_LEFT:
                printw("Left Arrow Pressed\n");
                break;
            case KEY_RIGHT:
                printw("Right Arrow Pressed\n");
                break;
            case 27: // 27 is the ASCII code for the Escape key (to exit the loop)
                endwin(); // Clean up ncurses before exiting
                return 0;
            default:
                break;
        }

        refresh(); // Refresh the screen to show the printed text
    }

    endwin(); // Clean up ncurses before exiting
    return 0;
}
