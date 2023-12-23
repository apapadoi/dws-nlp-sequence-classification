import sys

import pydub.exceptions
import requests
import json
import codecs
import datetime
from dateutil.parser import parse
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import fastf1
import speech_recognition as sr
from pydub import AudioSegment # NOTE: needs installation of ffmpeg
import librosa

if 'continue' in sys.argv:
    df = pd.read_csv('./data.csv')
    starting_year = 2022
else:
    df = pd.DataFrame([], columns=['text', 'label', 'features'])
    starting_year = 2018

base_url = 'https://livetiming.formula1.com/static/{}'
recognizer = sr.Recognizer()
audio_data_folder = './audio_data/'

if not os.path.exists(audio_data_folder):
    os.makedirs(audio_data_folder)

found_belgian_2022 = False
for year in range(starting_year, 2024):
    current_url = base_url.format(year) + '/Index.json'
    print(current_url)
    response = requests.get(current_url)
    decoded_response = response.content.decode('utf-8-sig')
    response_json = json.loads(decoded_response)
    response_initial = response_json
    for weekend in response_initial['Meetings']:
        if not found_belgian_2022 and year == 2022 and 'Belgian' not in [session['Path'].split('_')[1] for session in weekend['Sessions']]:
            continue

        found_belgian_2022 = True
        for session in weekend['Sessions']:
            if 'Path' not in session:
                print(f'No available path for {weekend["OfficialName"]}')
            else:
                response = requests.get(base_url.format(session['Path']) + 'Index.json')
                if response.status_code == 404:
                    print(f'Could not load data for {session["Path"]}')
                    continue

                decoded_response = response.content.decode('utf-8-sig')
                available_data_endpoints_json = json.loads(decoded_response)
                if 'TeamRadio' not in available_data_endpoints_json['Feeds']:
                    print(f'No available TeamRadio feed for {session["Path"]}')
                    continue

                # load TeamRadio
                print(f'Loading TeamRadio data for {session["Path"]}')
                team_radio_response = requests.get(base_url.format(session['Path'] + (available_data_endpoints_json['Feeds']['TeamRadio']['KeyFramePath'])))
                team_radio_decoded_response = team_radio_response.content.decode('utf-8-sig')
                team_radio_messages_list_json = json.loads(team_radio_decoded_response)

                if 'Captures' not in team_radio_messages_list_json:
                    print(f'No available TeamRadio feed for {session["Path"]}')
                    continue

                team_radio_captures = team_radio_messages_list_json['Captures']
                for capture in team_radio_captures:
                    current_radio_message_response = requests.get(base_url.format(session['Path'] + capture['Path']))
                    if current_radio_message_response.status_code == 200:
                        mp3_audio_file_path = audio_data_folder + capture['Path'].split('/')[1]
                        wav_audio_file_path = audio_data_folder + capture['Path'].split('/')[1].replace('.mp3', '.wav')

                        with open(mp3_audio_file_path, 'wb') as file:
                            file.write(current_radio_message_response.content)
                        try:
                            audio = AudioSegment.from_mp3(mp3_audio_file_path)
                        except pydub.exceptions.CouldntDecodeError:
                            print(f'Could not decode')
                            continue

                        audio_duration_seconds = len(audio) / 1000
                        print(f"{capture['Path']} audio duration: {audio_duration_seconds}")

                        audio.export(wav_audio_file_path, format="wav")

                        os.remove(mp3_audio_file_path)

                        text = ''
                        with sr.AudioFile(wav_audio_file_path) as source:
                            if audio_duration_seconds > 30:
                                for chunk in range(round(audio_duration_seconds / 30.0)):
                                    audio_data = recognizer.record(source, duration=30)
                                    try:
                                        transcript = recognizer.recognize_google(audio_data)
                                        text += transcript + " "
                                    except sr.UnknownValueError:
                                        print("Speech recognition could not understand audio")
                                    except sr.RequestError as e:
                                        print(f"Could not request results; {e}")
                            else:
                                audio_data = recognizer.record(source)
                                try:
                                    text = recognizer.recognize_google(audio_data)
                                except sr.UnknownValueError:
                                    print("Speech recognition could not understand audio")
                                except sr.RequestError as e:
                                    print(f"Could not request results; {e}")

                        y, sampling_rate = librosa.load(wav_audio_file_path)

                        stft = np.abs(librosa.stft(y))
                        average_stft = np.array([np.mean(row) for row in stft])

                        spectrogram = librosa.feature.melspectrogram(y=y, sr=sampling_rate)
                        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                        average_spectogram_db = np.array([np.mean(row) for row in spectrogram_db])

                        mfccs = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=13)
                        average_mfcss = np.array([np.mean(row) for row in mfccs])

                        # TODO might need to average per column - calculate total audio duration of dataset in seconds as well as total size of audio files collected
                        pitches, _ = librosa.piptrack(y=y, sr=sampling_rate)
                        average_pitches = np.array([np.mean(row) for row in pitches])

                        energy = librosa.feature.rms(y=y)
                        average_energy = np.array([np.mean(row) for row in energy])

                        all_features = np.concatenate((average_stft, average_spectogram_db, average_mfcss, average_pitches, average_energy), axis=0)
                        flattened_features = all_features.ravel()

                        df.loc[len(df)] = [text, wav_audio_file_path.split('_')[1][5:-2], ','.join([str(value) for value in flattened_features])]
                        df.to_csv('data.csv', index=False)
                        os.remove(wav_audio_file_path)
                    else:
                        print(f'Could not download {capture["Path"]}')
