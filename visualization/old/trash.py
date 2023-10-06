import time
from playsound import playsound

while True:

    playsound('sound.mp3')

    # Allow some time for the sound to finish playing
    time.sleep(1)  # Adjust the sleep time as needed based on your sound file's duration
