import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F

try:
    from pyctcdecode import build_ctcdecoder
    _has_pyctc = True
    print(" Using pyctcdecode for beam search")
except Exception:
    _has_pyctc = False
    print(" pyctcdecode not available. Beam search disabled.")

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode="max", blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.blank_id = blank_id
        self.search_mode = search_mode.lower()

        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.vocab = vocab

        if _has_pyctc and self.search_mode != "max":
            try:
                self.beam_decoder = build_ctcdecoder(self.vocab)
                print(" Beam decoder initialized")
            except Exception as e:
                print(" Beam init failed:", e)
                self.beam_decoder = None
        else:
            self.beam_decoder = None

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max" or self.beam_decoder is None:
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(dim=-1)
        nn_output = nn_output.detach().cpu().numpy()
        vid_lgt = vid_lgt.cpu().numpy()

        results = []
        for b in range(nn_output.shape[0]):
            L = int(vid_lgt[b])
            logit = nn_output[b][:L]
            try:
                decoded = self.beam_decoder.decode(logit)
            except Exception as e:
                print(" beam error:", e)
                return self.MaxDecode(torch.tensor(nn_output), torch.tensor(vid_lgt))

            # unicode -> class_id
            class_ids = [ord(ch) - 20000 for ch in decoded]
            sent = [(self.i2g_dict.get(cid, "UNK"), i)
                    for i, cid in enumerate(class_ids) if cid != self.blank_id]
            results.append(sent)
        return results

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:int(vid_lgt[batch_idx])])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
