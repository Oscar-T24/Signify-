import yt_dlp
import pandas as pd
from bs4 import BeautifulSoup
from data_analysis import main
import os
from main import LoadCV

def download_video(url, output_path="videos"):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_title = info.get('title', 'unknown')
        video_ext = info.get('ext', 'mkv')  # Default to mp4 if extension is not found

        # Construct the file path
        video_path = os.path.join(output_path, f"{video_title}.{video_ext}")


    return video_path

def extract_yt_videos():
    liste = []
    pandas = pd.read_csv("asl_database.csv")
    urls = pandas[("YouTube Video")].values
    for url in urls:
        if type(url)==float:  # Skip empty strings
            continue
        soup = BeautifulSoup(url, 'html.parser')
        iframe = soup.find('iframe').get('src')
        path = download_video(iframe)
        print("path to open",path)
        perc = main(download_video(iframe))
        print("finished processing video : rate =",perc)
        liste.append((path,perc))

    print(liste)


def analyze():
    """Processes the input video frame-by-frame and records hand landmarks."""
    for video_path in os.listdir("videos"):
        if video_path.endswith(".mkv"):
            # process video
            video = LoadCV("videos/"+video_path)  # Initialize video capture and hand tracking

            frame_counter = 0  # Keeps track of the frame index

            while True:
                ret, frame = video.cap.read()
                if not ret:
                    break  # Stop processing when the video ends

                video.record(counter=frame_counter)  # Record hand landmarks for this frame
                frame_counter += 1  # Increment frame index

            video.release()  # Release resources
            print(f"Finished analyzing {video_path}. Recorded {frame_counter} frames.")

analyze()
