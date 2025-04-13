#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
import speech_recognition as sr
import os

import webbrowser
#use in say function
import win32com.client
import datetime
#use to open a[[
import subprocess
#use to load weather api
import pyowm
#emial
import smtplib
from email.mime.text import MIMEText




def send_email(subject,message,to_email):
    from_email = 'i222591@nu.edu.pk'
    password = '2250154*'

    msg = MIMEText(message)
    msg['subject'] = subject
    msg['from_email'] = subject
    msg['to_email'] = subject

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)                           #smyp = simplemail transfer protocal
        server.starttls()
        server.login(from_email,password)
        server.sendmail(from_email,to_email,msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


speaker = win32com.client.Dispatch('SAPI.SpVoice')

owm = pyowm.OWM('c755420aa15fd3ff41b5ae801dbd8509')
def get_weather(city):
    try:
        observation = owm.weather_manager().weather_at_place(city)
        weather = observation.weather
        status = weather.status
        temperature = weather.temperature('celsius')['temp']
        print(f"The weather in {city} is {status} with a temperature of {temperature}°C.")
        return f"The weather in {city} is {status} with a temperature of {temperature}°C."
    except pyowm.exceptions.api_response_error.NotFoundError:
        return "Sorry, I couldn't find weather information for that city."

def say(text):
    speaker.Speak(text)

def Microphone_Voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #r.pause_threshold =1
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language='en-in')
            print(f'User said: {query}')
            return query
        except sr.UnknownValueError:
            print('Sorry, I could not understand what you said.')
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")


say('Hi I am Zubair Assistant , zubi')
while 1:
    print('Listening...')
    take_voice = Microphone_Voice()
    #say(take_voice)
    sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"],
             ["google", "https://www.google.com"], ]
    for site in sites:
        if f"Open {site[0]}".lower() in take_voice.lower():
            say(f"Opening {site[0]} sir...")
            webbrowser.open(site[1])

    if "open music" in take_voice:
        say('Open Music sir...')
        musicPath = r'C:\Users\Zubair\Downloads\Unison-Aperture-NCS-Release.mp3'
        os.system(f"start {musicPath}")

    elif "the time" in take_voice:
        hour = datetime.datetime.now().strftime("%H")
        min = datetime.datetime.now().strftime("%M")
        say(f"Sir time is {hour} bajke {min} minutes")
#"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe"

    elif "open app".lower() in take_voice.lower():
        say('sure sir')
        app_path = r"C:\Program Files\JetBrains\PyCharm Community Edition 2023.1.4\bin\pycharm64.exe"
        try:
            subprocess.Popen([app_path])
        except FileNotFoundError:
            print("File not found or unable to open the application.")

    elif "open editing".lower() in take_voice.lower():
        say('sure sir')
        app_path = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe"
        try:
            subprocess.Popen([app_path])
        except FileNotFoundError:
            print("File not found or unable to open the application.")

    elif "tell me about weather".lower() in take_voice.lower():
        say("Sure, which city's weather would you like to know?")
        city = Microphone_Voice()
        weather_info = get_weather(city)

        say(weather_info)



    ''' elif "send email" in take_voice:
        say("Sure, what should be the subject of the email?")
        email_subject = Microphone_Voice()
        say("What message would you like to send?")
        email_message = Microphone_Voice()
        #say("To whom should I send the email?")
        #email_recipient = Microphone_Voice()
        email_recipient = 'qq8955694@gmail.com'
#qq8955694
        if send_email(email_subject, email_message, email_recipient):
            say("Email sent successfully.")
        else:
            say("Sorry, I couldn't send the email.")'''






