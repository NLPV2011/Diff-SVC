from infer_tools.conversion import Svc
from utils.tools import get_configs_of
import numpy
import soundfile
import librosa
from infer_tools.slicer import Slicer
import torch
from tqdm import tqdm

def split(audio, sample_rate, hop_size, db_thresh = -24, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result

def cross_fade(a: numpy.ndarray, b: numpy.ndarray, idx: int):
    result = numpy.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    numpy.copyto(dst=result[:idx], src=a[:idx])
    k = numpy.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    numpy.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result

configs = get_configs_of("base_ds")
speaker_id = 0
key = 0
model_path = "output/ckpt/12000.ckpt"
model = Svc(configs=configs, model_path=model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

wav_path = "tuyam.wav"

sample_rate = 44100
hop_len = 512
x, sr = librosa.load(wav_path, sr=sample_rate)
if len(x.shape) > 1:
    x = librosa.to_mono(x)

result = numpy.zeros(0)
current_length = 0
segments = split(x, sample_rate, hop_len)
print('Cut the input audio into ' + str(len(segments)) + ' slices')

output_sample_rate=44100
with torch.no_grad():
    for segment in tqdm(segments):
        start_frame = segment[0]
        seg_input = segment[1]
        seg_output, output_sample_rate = model.infer_segment(wav=seg_input, sr=sr, speaker_id=speaker_id, trans=key)
        
        silent_length = round(start_frame * hop_len * output_sample_rate / sample_rate) - current_length
        if silent_length >= 0:
            result = numpy.append(result, numpy.zeros(silent_length))
            result = numpy.append(result, seg_output)
        else:
            result = cross_fade(result, seg_output, current_length + silent_length)
        current_length = current_length + silent_length + len(seg_output)
    #sf.write(cmd.output, result, output_sample_rate)
    soundfile.write("dash_0_0_0_0_0.wav", result, output_sample_rate)

# output, sr = model.infer_segment(wav=x, sr=sr, speaker_id=speaker_id, trans=key)
