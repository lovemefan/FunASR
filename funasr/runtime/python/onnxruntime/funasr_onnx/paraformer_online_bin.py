# -*- coding:utf-8 -*-
# @FileName  :paraformer_online_bin.py
# @Time      :2023/4/29 02:03
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import copy
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np

from funasr.utils.postprocess_utils import sentence_postprocess
from .paraformer_bin import Paraformer
from .utils.timestamp_utils import time_stamp_lfr6_onnx


class ParaformerOnline(Paraformer):
    def __init__(self,
                 model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 plot_timestamp_to: str = "",
                 quantize: bool = False,
                 intra_op_num_threads: int = 4,
                 **kwargs):
        super().__init__(model_dir, batch_size, device_id, plot_timestamp_to, quantize, intra_op_num_threads)

    def __call__(self, waveform: Union[str, np.ndarray, List[str]], **kwargs):
        waveform_list = self.load_data(waveform, self.frontend.opts.frame_opts.samp_freq)
        feats, feats_len = self.extract_feat(waveform_list)

        param_dict = kwargs.get('param_dict', dict())
        in_cache = param_dict.get('in_cache', list())
        in_cache = self.prepare_cache(in_cache)
        asr_res = []
        outputs = self.infer(feats.astype(np.float32), in_cache)

        am_scores, valid_token_lens = outputs[0], outputs[1]
        if len(outputs) == 4:
            # for BiCifParaformer Inference
            us_alphas, us_peaks = outputs[2], outputs[3]
        else:
            us_alphas, us_peaks = None, None
        preds = self.decode(am_scores, valid_token_lens)

        if us_peaks is None:
            for pred in preds:
                pred = sentence_postprocess(pred)
                asr_res.append({'preds': pred})
        else:
            for pred, us_peaks_ in zip(preds, us_peaks):
                raw_tokens = pred
                timestamp, timestamp_raw = time_stamp_lfr6_onnx(us_peaks_, copy.copy(raw_tokens))
                text_proc, timestamp_proc, _ = sentence_postprocess(raw_tokens, timestamp_raw)
                # logging.warning(timestamp)
                if len(self.plot_timestamp_to):
                    self.plot_wave_timestamp(waveform_list[0], timestamp, self.plot_timestamp_to)
                asr_res.append({'preds': text_proc, 'timestamp': timestamp_proc, "raw_tokens": raw_tokens})

        # todo validate result
        print(asr_res)
        return asr_res

    def infer(self, feats: np.ndarray,
              in_cache) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats] + in_cache)
        return outputs

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        output_size = self.encoder_conf["output_size"]
        cache_num = self.decoder_conf["att_layer_num"]
        kernel_size = self.decoder_conf["kernel_size"]

        in_cache = [
            np.zeros((1, output_size, kernel_size), dtype=np.float32)
            for _ in range(cache_num)
        ]
        return in_cache