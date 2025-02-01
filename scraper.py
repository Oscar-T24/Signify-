import yt_dlp
import pandas as pd

def download_video(url, output_path="."):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading video from {url}...")
        ydl.download([url])
        print("Download completed!")

def extract_yt_videos():
    pandas = pd.open_csv("asl_database.csv")

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/embed/ys0QCfNZWZc'
    download_video(video_url)
