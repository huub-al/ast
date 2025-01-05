# 4 - 1 - 2025
# Huub Al
# python script that generates spectograms based on song id.

import requests
import os
import sys
import json
import io
import torchaudio
from bs4 import BeautifulSoup
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from egs.audioset.inference import make_features, load_label
from src.models import ASTModel

# Step 1: Scrape Spotify embed for song details
def get_song_info_from_embed(song_id):
    # Spotify Embed URL pattern
    embed_prefix = "https://open.spotify.com/embed/track/"

    embed_url = f"{embed_prefix}{song_id}"
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
def generate_spectogram(preview_url):
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

    return make_features(audio, mel_bins=128)
        

if __name__ == "__main__":
    print("--- Spotify Sound predictor.---")

    songs_with_url = [
        '3DK6m7It6Pw857FcQftMds', # runaway kanye west
        '7a1XGxwYRr3YUW3NpKrBDX', # birds in a forest
        '3GCdLUSnKSMJhs4Tj6CV3s', # all the stars 
        '186bI2lE3fYyyuCaXtn7Ic', # car sounds
        ]

    # define path and settings    
    pretrained_mdl_path = '../pretrained_models/audioset_10_10_0.4593.pth'
    fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])

    # initialize an AST model
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    sd = torch.load(pretrained_mdl_path, map_location=device, weights_only=True)
    audio_model = ASTModel(input_tdim=1024, fstride=fstride, tstride=tstride,
                            imagenet_pretrain=False, audioset_pretrain=False)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # 4. map the post-prob to label
    label_csv = '../egs/audioset/data/class_labels_indices.csv'
    labels = load_label(label_csv)

    for song in songs_with_url:
        song_info = get_song_info_from_embed(song)

        if song_info:
            print("\n--- Song Information ---")
            print(f"Name: {song_info['name']}")
            print(f"Artist: {song_info['artist']}")
            print(f"Preview URL: {song_info['preview_url']}")

            #generate spectogram
            feats = generate_spectogram(song_info['preview_url'])
            input_tdim = feats.shape[0]

            # 3. feed the data feature to model
            feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature

            audio_model.eval()                                      # set the eval model
            with torch.no_grad():
                output = audio_model.forward(feats_data)
                output = torch.sigmoid(output)
            result_output = output.data.cpu().numpy()[0]

            sorted_indexes = np.argsort(result_output)[::-1]

            # Print audio tagging top probabilities
            print('[*INFO] predice results:')
            for k in range(10):
                print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                        result_output[sorted_indexes[k]])) 

        else:
            print("No song found with the given ID or an error occurred.")

