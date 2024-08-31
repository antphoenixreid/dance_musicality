from pytube import YouTube, Search
from moviepy.editor import *

import os
import pandas as pd

home_dir = 'E:/Engineering/Signal Processing/Personal Projects/Dance Musicality/'
file_path = os.path.join(home_dir, 'Data')
music_clip_fn = os.path.join(file_path, 'Music_Clips.xlsx')

# Read all the links from the Excel file
rbdys_data = pd.read_excel(music_clip_fn, header=0, sheet_name='Red Bull Dance Your Style 2022')

rbdys_song_list = []
for ind, row in rbdys_data.iterrows():
    dancer = row['Dancer']
    song = row['Song']

    song_link = row['YouTube Link']
    yt_link = YouTube(song_link).streams.get_highest_resolution()

    path = os.path.join(file_path, 'Red Bull Dance Your Style 2022', dancer)

    isDir = os.path.exists(path)
    if not isDir:
        os.mkdir(path)

    try:    
        stream = yt_link.download(output_path=path)
        
        base, ext = os.path.splitext(stream)
        mp3_file = base + '.mp3'
        
        video_obj = VideoFileClip(stream)

        audio = video_obj.audio

        audio.write_audiofile(mp3_file)

        os.remove(stream)
    except:
        if os.path.exists(stream):
            os.remove(stream)