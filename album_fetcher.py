from spotipy.oauth2 import SpotifyOAuth
import spotipy
import requests
from PIL import Image
from io import BytesIO
import time

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="",
    client_secret="",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-currently-playing"
))

def save_album_art():
    try:
        current = sp.current_playback()
        if current and current['item']:
            url = current['item']['album']['images'][0]['url']
            response = requests.get(url)
            img_pil = Image.open(BytesIO(response.content)).resize((60, 60))
            img_pil.save("current_album.jpg")
    except Exception as e:
        print("Error fetching album art:", e)

while True:
    save_album_art()
    time.sleep(5)  # Update every 5 seconds
