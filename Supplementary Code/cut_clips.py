import os
from moviepy.editor import *
import pandas as pd

def extract_clips(video_file, clip_duration, clip_start_times):
    clip_list = []
    with VideoFileClip(video_file) as video:
        for i in range(len(clip_start_times)):
            clip = video.subclip(clip_start_times[i], clip_start_times[i] + clip_duration[i])
            clip_list.append(clip)
    return clip_list

# Opening the CSV file
home_dir = 'E:/Engineering/Signal Processing/Personal Projects/Dance Musicality/'
file_path = os.path.join(home_dir, 'Data')
music_clip_fn = os.path.join(file_path, 'Music_Clips.xlsx')
rbdys_video_fn = os.path.join(file_path, 'Red Bull Dance Your Style 2022', 'RBDYS_2022.mp4')
rbdys_video = VideoFileClip(rbdys_video_fn)

# For each clip information, set the start time (in seconds) and the duration of the clip (start time - end time; in seconds)
# Save in Dictionary
# Read Red Bull Dance Your Style 2022
rbdys_data = pd.read_excel(music_clip_fn, header=0, sheet_name='Red Bull Dance Your Style 2022')

for ind, row in rbdys_data.iterrows():
    dancer = row['Dancer']

    path = os.path.join(file_path, 'Red Bull Dance Your Style 2022', dancer)
    clip_name = dancer + '_clip.mp4'
    
    isDir = os.path.exists(path)
    if not isDir:
        os.mkdir(path)

    start_time = row['Start']
    start_time = start_time.hour*60 + start_time.minute

    end_time = row['End']
    end_time = end_time.hour*60 + end_time.minute

    clip = rbdys_video.subclip(start_time, end_time)
    
    clip_path = os.path.join(path, clip_name)
    clip.write_videofile(clip_path)
    