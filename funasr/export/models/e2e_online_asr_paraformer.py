import logging
import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.models.encoder.sanm_encoder import SANMEncoder
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.export.models.encoder.sanm_encoder import SANMEncoder as SANMEncoder_export
from funasr.export.models.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder_export
from funasr.models.predictor.cif import CifPredictorV2, CifPredictorV3
from funasr.export.models.predictor.cif import CifPredictorV2 as CifPredictorV2_export
from funasr.export.models.predictor.cif import CifPredictorV3 as CifPredictorV3_export
from funasr.models.decoder.sanm_decoder import ParaformerSANMDecoder
from funasr.models.decoder.transformer_decoder import ParaformerDecoderSAN
from funasr.export.models.decoder.sanm_decoder import ParaformerSANMDecoder as ParaformerSANMDecoder_export
from funasr.export.models.decoder.transformer_decoder import ParaformerDecoderSAN as ParaformerDecoderSAN_export


class ParaformerOnline(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.encoder, SANMEncoder):
            self.encoder = SANMEncoder_export(model.encoder, onnx=onnx)
        elif isinstance(model.encoder, ConformerEncoder):
            self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        if isinstance(model.predictor, CifPredictorV2):
            self.predictor = CifPredictorV2_export(model.predictor)
        if isinstance(model.decoder, ParaformerSANMDecoder):
            self.decoder = ParaformerSANMDecoder_export(model.decoder, onnx=onnx)
        elif isinstance(model.decoder, ParaformerDecoderSAN):
            self.decoder = ParaformerDecoderSAN_export(model.decoder, onnx=onnx)

        self.feats_dim = feats_dim
        self.model_name = model_name

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

    def forward(
            self,
            speech: torch.Tensor,
            in_cache
    ):

        cache = self._prepare_cache()
        speech_lengths = speech.shape[1]
        enc, enc_len = self.encoder(speech, speech_lengths, cache=cache["encoder"])

        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, None)
        pre_token_length = pre_token_length.floor().type(torch.int32)

        decoder_out, memory, in_cache = self.decoder.forward_chunk(enc, pre_acoustic_embeds, in_cache)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)

        return decoder_out, pre_token_length, in_cache

    def _prepare_cache(self, cache: dict = None, chunk_size=[5,10,5], batch_size=1):

        enc_output_size = 320
        feats_dims = 80 * 7
        cache = cache or dict()
        cache_en = {"start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
                    "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size, "last_chunk": False,
                    "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)), "tail_chunk": False}
        cache["encoder"] = cache_en
        cache["encoder"]["is_final"] = False

        cache_de = {"decode_fsmn": None}
        cache["decoder"] = cache_de

        return cache

    def _cache_reset(cache: dict = {}, chunk_size=[5,10,5], batch_size=1):
        if len(cache) > 0:

            enc_output_size = 320
            feats_dims = 80 * 7
            cache_en = {"start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
                        "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size, "last_chunk": False,
                        "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)), "tail_chunk": False}
            cache["encoder"] = cache_en

            cache_de = {"decode_fsmn": None}
            cache["decoder"] = cache_de

        return cache

    def get_dummy_inputs(self):
        speech = torch.randn(1, 60, self.feats_dim)

        cache_num = len(self.decoder.model.decoders)
        cache = [
            torch.zeros((1, self.decoder.model.decoders[0].size, self.decoder.model.decoders[0].self_attn.kernel_size))
            for _ in range(cache_num)
        ]

        return (speech, cache)

    def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
        import numpy as np
        fbank = np.loadtxt(txt_file)
        fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
        speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
        speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
        return (speech,)

    def get_input_names(self):
        cache_num = len(self.decoder.model.decoders)

        return ['speech'] + ['in_cache_%d' % i for i in range(cache_num)]

    def get_output_names(self):
        cache_num = len(self.decoder.model.decoders)
        return ['logits', 'token_num'] + ['out_cache_%d' % i for i in range(cache_num)]

    def get_dynamic_axes(self):
        cache_num = len(self.decoder.model.decoders)

        ret = {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },

            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }

        ret.update({
            'in_cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                2: 'cache_%d_length' % d
            }
            for d in range(cache_num)
        })

        ret.update({
            'out_cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                2: 'cache_%d_length' % d
            }
            for d in range(cache_num)
        })
        return ret