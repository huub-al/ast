import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import io
import torchaudio
from bs4 import BeautifulSoup

# Spotify Embed URL pattern
EMBED_URL = "https://open.spotify.com/embed/track/"

# Step 1: Scrape Spotify embed for song details
def get_song_info_from_embed(song_id):
    embed_url = f"{EMBED_URL}{song_id}"
    response = requests.get(embed_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the script tag containing the JSON
        script_tag = soup.find('script', string=lambda string: string and 'props' in string)
        if script_tag:
            json_text = script_tag.string.strip()
            json_data = json.loads(json_text)

            # Navigate to the required fields
            entity = json_data['props']['pageProps']['state']['data']['entity']

            return {
                'name': entity['name'],
                'artist': entity['artists'][0]['name'],
                'preview_url': entity['audioPreview'].get('url')
            }
        else:
            print("Could not find the script tag containing JSON data.")
            return None
    else:
        print(f"Failed to retrieve embed page. Status code: {response.status_code}")
        return None

# Step 2: Generate and plot a spectrogram from an audio preview URL
def plot_spectrogram_from_preview(preview_url):
    if not preview_url:
        print("No preview URL provided.")
        return

    # Download the audio preview
    print("Downloading audio preview...")
    response = requests.get(preview_url)

    if response.status_code == 200:
        print("Processing audio data...")
        # Load audio from raw data
        audio, sr = torchaudio.load(io.BytesIO(response.content))

        # check sample rate
        if sr != 16e3:
            audio = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16e3
                )(audio) 

        print(audio.shape)
        audio = np.array(audio.mean(axis=0))
        print(audio.shape)

        # Generate spectrogram
        print("Generating spectrogram...")
        spectrogram = librosa.stft(audio, n_fft=2048)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (log(Hz))")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Failed to download audio preview. Status code: {response.status_code}")

if __name__ == "__main__":
    print("--- Spotify Song Info Retriever ---")

    songs_with_url = [
        '2BOUrjXoRIo2YHVAyZyXVX',
        '43Vb1wOGyRfk7fsZ5ZLRH8',
        '0TCC2Kwusv749hTFrO7d9Q',
        '3DK6m7It6Pw857FcQftMds'
        ]

    for song in songs_with_url:

        song_info = get_song_info_from_embed(song)

        if song_info:
            print("\n--- Song Information ---")
            print(f"Name: {song_info['name']}")
            print(f"Artist: {song_info['artist']}")
            print(f"Preview URL: {song_info['preview_url']}")

            # Plot spectrogram from preview URL
            plot_spectrogram_from_preview(song_info['preview_url'])
        else:
            print("No song found with the given ID or an error occurred.")