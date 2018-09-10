from os import listdir
from os.path import join, isdir
import os
from shutil import copyfile
import subprocess
import re

from sklearn.model_selection import train_test_split


def preprocess():
    # checkLength()
    # strechAudio()
    # distribute_emotion()
    # distributeTrainTest()
    normalizedAudio()
    # source2Chunks()

def strechAudio():
    source_dir = "dataset"
    target_dir = "dataset_stretch"
    clip_dir = "dataset_stretch_clip"


    for attr in ['Happy', 'Sad', 'Neutral', 'Angry']:
        for d in [target_dir, clip_dir]:
            if not isdir(join(d, attr)):
                os.mkdir(join(d, attr))
        for audio_name in listdir(join(source_dir, attr)):
            get_length_command = "ffmpeg -i {} 2>&1 | grep Duration".format(
                join(source_dir, attr, audio_name))
            out = subprocess.check_output(get_length_command, shell=True)
            out = out.decode()
            length = re.findall(r"Duration: \d\d:\d\d:\d(\d.\d\d)", out)[0]
            stretch_command = "ffmpeg -i {0} -codec:a libmp3lame -filter:a \"atempo=sqrt({1}),atempo=sqrt({1})\" -b:a 16K {2}".format(
                join(source_dir, attr, audio_name), length, join(target_dir, attr, audio_name)
            )
            subprocess.run(stretch_command, shell=True)
            clip_command = "ffmpeg -i {} -ss 0 -t 1.00 -acodec copy {}".format(
                join(target_dir, attr, audio_name), join(clip_dir, attr, audio_name))
            subprocess.run(clip_command, shell=True)


def checkLength():
    clip_dir = "dataset_stretch_clip"

    for attr in ['Happy', 'Sad', 'Neutral', 'Angry']:
        for audio_name in listdir(join(clip_dir, attr)):
            get_length_command = "ffmpeg -i {} 2>&1 | grep Duration".format(
                join(clip_dir, attr, audio_name))
            out = subprocess.check_output(get_length_command, shell=True)
            out = out.decode()
            length = re.findall(r"Duration: \d\d:\d\d:\d(\d.\d\d)", out)[0]
            # print(length)
            if length != "1.01":
                print(join(attr, audio_name), length)

    

def distribute_emotion():
    emotions = ["anger", "boredom", "disgust",
                "fear", "happiness", "sadness", "neural"]
    letters = ["W", "L", "E", "A", "F", "T", "N"]
    emotion_dict = dict(zip(letters, emotions))
    print(emotion_dict)
    for emotion in emotions:
        if not isdir(emotion):
            os.mkdir(emotion)

    source_dir = "wav"
    for audio in listdir(source_dir):
        tag = audio[5]
        try:
            target_dir = emotion_dict[tag]
            copyfile(join(source_dir, audio), join(target_dir, audio))
        except KeyError:
            pass


def distributeTrainTest():
    dir_names = ["train", "test"]
    for dir_name in dir_names:
        if not isdir(dir_name):
            os.mkdir(dir_name)

    emotions = ["anger", "boredom", "disgust",
                "fear", "happiness", "sadness", "neural"]
    train_x_list = []
    train_y_list = []

    test_x_list = []
    test_y_list = []

    for emotion in emotions:
        audios = listdir(emotion)
        # print(len(audios))
        x_train, x_test, y_train, y_test = train_test_split(audios,
                                                            [emotion for _ in range(len(audios))], test_size=0.2)
        # print(len(x_train), x_train)
        # print(len(x_test), x_test)
        for attr, dir_n, x_list, y_list in zip([x_train, x_test], dir_names,
                                               [train_x_list, test_x_list], [train_y_list, test_y_list]):
            for audio in attr:
                copyfile(join(emotion, audio), join(dir_n, audio))
                x_list.append(audio.split(".")[0])
                y_list.append(emotion)
            # print(dict_name)

    train_df = pd.DataFrame({"x": train_x_list, "y": train_y_list})
    test_df = pd.DataFrame({"x": test_x_list, "y": test_y_list})
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)


def removeSilence():
    """ Remove the silent part of audio
    Ref:
        - https://stackoverflow.com/questions/25697596/using-ffmpeg-with-silencedetect-to-remove-audio-silence
    """
    source_dir = "dataset"
    target_dir = "dataset_silence_remove"

    for attr in ['Happy', 'Sad', 'Neutral', 'Angry']:
        if not isdir(join(target_dir, attr)):
            os.mkdir(join(target_dir, attr))
        for audio_name in listdir(join(source_dir, attr)):
            command = "ffmpeg -i {} -af silenceremove=1:0:-50dB {}".format(
                join(source_dir, attr, audio_name),
                join(target_dir, attr, audio_name))
            subprocess.run(command, shell=True)


def normalizedAudio():
    """ Normalized audio with ffmpeg
    Ref:
        - https://superuser.com/questions/323119/how-can-i-normalize-audio-using-ffmpeg
        - https://trac.ffmpeg.org/wiki/AudioVolume
    """
    source_dir = "dataset_stretch_clip_remove"
    target_dir = "dataset_stretch_clip_remove_norm"

    for attr in ['Happy', 'Sad', 'Neutral', 'Angry']:
        if not isdir(join(target_dir, attr)):
            os.mkdir(join(target_dir, attr))
        for audio_name in listdir(join(source_dir, attr)):
            command = "ffmpeg -i {} -filter:a loudnorm {}".format(
                join(source_dir, attr, audio_name),
                join(target_dir, attr, audio_name))
            subprocess.run(command, shell=True)


def source2Chunks():
    """ Split audio to 20ms-length chunks
    Ref:
        - https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple
    """
    source_dir = "dataset_silence_remove_norm"
    target_dir = "dataset_silence_remove_norm_20"
    chunk_length = 0.02

    for dic_name in ['Happy', 'Sad', 'Neutral', 'Angry']:
        for audio_name in listdir(join(source_dir, dic_name)):
            #print(audio_name)
            if not isdir(join(target_dir, dic_name)):
                os.mkdir(join(target_dir, dic_name))
            audio_prefix = audio_name.split(".")[0]
            command = "ffmpeg -i {} -f segment -segment_time {} -c copy {}".format(
                join(source_dir, dic_name,audio_name), chunk_length,
                join(target_dir, dic_name, "{}_%03d.wav".format(audio_prefix)))
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    preprocess()
