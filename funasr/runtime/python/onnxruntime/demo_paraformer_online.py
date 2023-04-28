# -*- coding:utf-8 -*-
# @FileName  :demo_paraformer_online.py
# @Time      :2023/4/29 02:02
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com


import soundfile
from funasr_onnx.paraformer_online_bin import ParaformerOnline

model_dir = "/home/zlf/FunASR/onnx_export/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"
speech, sample_rate = soundfile.read("/dataset/speech/aishell/data_aishell/wav/test/S0764/BAC009S0764W0424.wav")
speech_length = speech.shape[0]
model = ParaformerOnline(model_dir, batch_size=2, plot_timestamp_to="./", pred_bias=0)  # cpu

sample_offset = 0

#[5, 10, 5] 600ms, [8, 8, 4] 480ms
chunk_size = [8, 8, 4]
stride_size = chunk_size[1] * 960
param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
final_result = ""

for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
    if sample_offset + stride_size >= speech_length - 1:
        stride_size = speech_length - sample_offset
        param_dict["is_final"] = True
    rec_result = model(speech[sample_offset: sample_offset + stride_size], param_dict=param_dict)
    if len(rec_result) != 0:
        final_result += rec_result['text'][0]
        print(rec_result, final_result)
print(final_result)