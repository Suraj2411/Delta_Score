import time
from collections import defaultdict

import torch
# import pandas as pd

from transformers import AutoTokenizer

from utils import (score_calc, get_idf_dict, get_model, get_tokenizer,lang2model, model2layers)

def delta_score(cands, refs, model_type=None, num_layers=None, verbose=False, idf=False, device=None, batch_size=64, nthreads=4, all_layers=False, lang=None, use_fast_tokenizer=False):
    
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert ( lang is not None or model_type is not None), "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
        
    start = time.perf_counter()
    all_preds = score_calc(model, refs, cands, tokenizer, idf_dict, verbose=verbose, device=device, batch_size=batch_size).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)


    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(
            f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
        )

    return out

