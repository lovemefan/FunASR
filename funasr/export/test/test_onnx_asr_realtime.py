import onnxruntime
import numpy as np

if __name__ == '__main__':
    onnx_path = "/home/zlf/FunASR/onnx_export/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/model.onnx"
    sess = onnxruntime.InferenceSession(onnx_path)
    input_name = [nd.name for nd in sess.get_inputs()]
    output_name = [nd.name for nd in sess.get_outputs()]

    def _get_feed_dict(feats_length):
        cache = [
            np.zeros((1, 320, 11), dtype=np.float32)
            for _ in range(12)
        ]
        result = {'speech': np.zeros((1, feats_length, 560), dtype=np.float32)
                 }
        result.update({
            'in_cache_%d' % d: cache[d]
            for d in range(12)
        })
        return result


    def _run(feed_dict):
        output = sess.run(output_name, input_feed=feed_dict)
        for name, value in zip(output_name, output):
            print('{}: {}'.format(name, value.shape))


    _run(_get_feed_dict(30))
    _run(_get_feed_dict(200))