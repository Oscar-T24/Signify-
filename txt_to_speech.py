from gtts import gTTS
import playsound
import os

# Text you want to convert to speech
#text = "Hello, I am your assistant speaking directly."
def speech(text):
# Choose language (English in this case)


    # Create a TTS object
    tts= gTTS(text=text, lang='en', slow=False)

    # Save the speech to a temporary file
    tts.save("temp.mp3")     

    # Play the speech (without saving permanently)
    playsound.playsound("temp.mp3")
    os.remove("temp.mp3")
