# Initialize my_flags array
my_flags = [False] * 10

# Open the settings.cfg file for reading
with open('settings.cfg', 'r') as file:

    index = 0
    # Read each line in the file
    for line in file:
        # Split the line into parts based on ':'
        parts = line.strip().split(':')
        
        # Get the index and values after ':'
        values = parts[1].split(',')

        if index < 3:
            # Update the my_flags array based on the values
            for i, value in enumerate(values):
                if 'f' in value:
                    my_flags[index*3] = True
                if 'm' in value:
                    my_flags[index*3 + 1] = True
                if 's' in value:
                    my_flags[index*3 + 2] = True
        else:
            for i, value in enumerate(values):
                if '1' in value:
                    my_flags[-1] = True
        index+=1



# Print the resulting my_flags array
print(my_flags)