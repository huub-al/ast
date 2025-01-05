import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def main():
    # Ensure the script has the correct arguments
    if len(sys.argv) != 3:
        print("Usage: python db_gen.py <.env_file> <N>")
        sys.exit(1)

    env_file = sys.argv[1]
    max_songs = int(sys.argv[2])

    # Load environment variables from the .env file
    load_dotenv(env_file)

    CLIENT_ID = os.getenv("CLIENTID")
    CLIENT_SECRET = os.getenv("CLIENTSECRET")

    if not CLIENT_ID or not CLIENT_SECRET:
        print("CLIENTID or CLIENTSECRET not found in the .env file.")
        sys.exit(1)

    # Set up Spotipy authentication
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri="http://localhost:8888/callback",
        scope="user-library-read"
    ))

    # Get the user's Spotify username
    user_profile = sp.current_user()
    username = user_profile["id"]

    # Retrieve liked songs
    liked_songs = []
    results = sp.current_user_saved_tracks(limit=50)  # Spotify API max limit per request is 50
    while results and len(liked_songs) < max_songs:
        for item in results["items"]:
            if len(liked_songs) >= max_songs:
                break

            track = item["track"]
            liked_songs.append({
                "id": track["id"],
                "title": track["name"],
                "artist": ", ".join(artist["name"] for artist in track["artists"]),
                "album": track["album"]["name"]
            })

        # Fetch the next batch of songs, if available
        results = sp.next(results) if results["next"] else None

    # Create a dictionary with indices as keys
    liked_songs_dict = {index: song for index, song in enumerate(liked_songs)}

    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)

    # Save to a JSON file in the data folder
    date_str = datetime.now().strftime("%Y%m%d")
    output_filename = os.path.join("data", f"{username}_liked_{len(liked_songs)}_{date_str}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(liked_songs_dict, f, indent=4, ensure_ascii=False)

    print(f"Liked songs saved to {output_filename}")

if __name__ == "__main__":
    main()
