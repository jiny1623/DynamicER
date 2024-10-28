# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# Copyright (c) 2024 Jinyoung Kim and authors of DynamicER.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import random
import math
import time
import torch
import numpy as np
from tqdm import tqdm
import pickle
import copy
from sklearn.cluster import KMeans

import blink.biencoder.data_process_mult as data_process
import blink.candidate_ranking.utils as utils
from blink.biencoder.biencoder import BiEncoderRanker
from blink.crossencoder.crossencoder import CrossEncoderRanker
from blink.common.params import BlinkParser

def load_data(data_split,
              bi_tokenizer,
              max_context_length,
              max_cand_length,
              knn,
              pickle_src_path,
              params,
              logger,
              return_dict_only=False):
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(pickle_src_path, 'entity_dictionary.pickle')
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, 'rb') as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True

    if return_dict_only and entity_dictionary_loaded:
        return entity_dictionary

    # Load data
    tensor_data_pkl_path = os.path.join(pickle_src_path, f'{data_split}_tensor_data.pickle')
    processed_data_pkl_path = os.path.join(pickle_src_path, f'{data_split}_processed_data.pickle')
    if os.path.isfile(tensor_data_pkl_path) and os.path.isfile(processed_data_pkl_path):
        print("Loading stored processed data...")
        with open(tensor_data_pkl_path, 'rb') as read_handle:
            tensor_data = pickle.load(read_handle)
        with open(processed_data_pkl_path, 'rb') as read_handle:
            processed_data = pickle.load(read_handle)
    else:
        data_samples = utils.read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(os.path.join(params["data_path"], 'dictionary.pickle'), 'rb') as read_handle:
                entity_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in data_samples[0].keys()
        # Filter samples without gold entities
        data_samples = list(
            filter(lambda sample: (len(sample["labels"]) > 0) if mult_labels else (sample["label"] is not None),
                   data_samples))
        logger.info("Read %d data samples." % len(data_samples))

        processed_data, entity_dictionary, tensor_data = data_process.process_mention_data(
            data_samples,
            entity_dictionary,
            bi_tokenizer,
            max_context_length,
            max_cand_length,
            context_key=params["context_key"],
            multi_label_key="labels" if mult_labels else None,
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            knn=knn,
            dictionary_processed=entity_dictionary_loaded
        )
        print("Saving processed data...")
        if not entity_dictionary_loaded:
            with open(entity_dictionary_pkl_path, 'wb') as write_handle:
                pickle.dump(entity_dictionary, write_handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        with open(tensor_data_pkl_path, 'wb') as write_handle:
            pickle.dump(tensor_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(processed_data_pkl_path, 'wb') as write_handle:
            pickle.dump(processed_data, write_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if return_dict_only:
        return entity_dictionary
    return entity_dictionary, tensor_data, processed_data


def save_topk_biencoder_cands(bi_reranker,
                             use_types, logger, n_gpu, params,
                             bi_tokenizer, max_context_length,
                             max_cand_length, pickle_src_path, topk=64):
    entity_dictionary = load_data('evaluate',
                                  bi_tokenizer,
                                  max_context_length,
                                  max_cand_length,
                                  1,
                                  pickle_src_path,
                                  params,
                                  logger,
                                  return_dict_only=True)

    if params["path_to_gold"]:
        train_tensor_data_pkl_path = os.path.join(
            params["path_to_gold"], "train_tensor_data.pickle"
        )
        train_processed_data_pkl_path = os.path.join(
            params["path_to_gold"], "train_processed_data.pickle"
        )
        assert (os.path.isfile(train_tensor_data_pkl_path) and
                os.path.isfile(train_processed_data_pkl_path)
        )
        with open(train_tensor_data_pkl_path, "rb") as read_handle:
            train_tensor_data = pickle.load(read_handle)
        with open(train_processed_data_pkl_path, "rb") as read_handle:
            train_processed_data = pickle.load(read_handle)

    elif params["path_to_prediction"]:
        test_mention_data_pkl_path = os.path.join(
            params["path_to_prediction"], "test_mention_data.pickle"
        )
        test_result_path = os.path.join(
            params["path_to_prediction"], params["pred_json"]
        )
        assert (os.path.isfile(test_mention_data_pkl_path) and
                os.path.isfile(test_result_path)
        )

        with open(test_mention_data_pkl_path, "rb") as read_handle:
            test_mention_data = pickle.load(read_handle)
        with open(test_result_path, "r") as f:
            test_result = json.load(f)
        
        pred_data = {}
        for item in test_mention_data:
            pred_data[item['mention_id']] = {"ids": item['context']['ids'], "predicted": item['label_cuis'][0]}

        for res in test_result['failure']:
            pred_data[res['mention_id']]['predicted'] = res['predicted_cui'] 
        
    entity_dict_vecs = torch.tensor(list(map(lambda x: x['ids'], entity_dictionary)), dtype=torch.long)

    if params["path_to_gold"]: 
        tensor_data = []

        for entry in tqdm(entity_dictionary, desc="continual case"):
            cui = entry['cui']

            matching_ids = []
            for item in train_processed_data:
                if cui in item['label_cuis']:
                    matching_ids.append(torch.tensor(item['context']['ids'], dtype=torch.long))

            # if len(matching_ids) > 30:
            #     matching_ids = random.sample(matching_ids, 30)

            tensor_data.append(matching_ids)

        entity_dict_vecs = [(e, t) for e, t in zip(entity_dict_vecs, tensor_data)]

    elif params["path_to_prediction"]:
        tensor_data = []
        
        for entry in tqdm(entity_dictionary, desc="prediction: continual case"):
            cui = entry['cui']

            matching_ids = []
            for mention_id, value in pred_data.items():
                if cui == value['predicted']:
                    matching_ids.append(torch.tensor(value['ids'], dtype=torch.long))

            tensor_data.append(matching_ids)

        entity_dict_vecs = [(e, t) for e, t in zip(entity_dict_vecs, tensor_data)]

    else:
        entity_dict_vecs = [(e, []) for e in entity_dict_vecs]

    logger.info('Biencoder: Embedding and indexing entity dictionary')
    if use_types:
        _, dict_indexes, dict_idxs_by_type = data_process.embed_and_index(
            bi_reranker, entity_dict_vecs, encoder_type="candidate", alpha=params["alpha"], n_gpu=n_gpu, corpus=entity_dictionary,
            force_exact_search=True, batch_size=params['embed_batch_size'])
    else:
        _, dict_index = data_process.embed_and_index(bi_reranker, entity_dict_vecs,
                                                     encoder_type="candidate", alpha=params["alpha"], n_gpu=n_gpu,
                                                     force_exact_search=True,
                                                     batch_size=params['embed_batch_size'])
    logger.info('Biencoder: Embedding and indexing finished')

    for mode in ["evaluate"]:
        logger.info(f"Biencoder: Fetching top-{topk} biencoder candidates for {mode} set")
        _, tensor_data, processed_data = load_data(mode,
                                                   bi_tokenizer,
                                                   max_context_length,
                                                   max_cand_length,
                                                   1,
                                                   pickle_src_path,
                                                   params,
                                                   logger)
        men_vecs = tensor_data[:][0]

        logger.info('Biencoder: Embedding mention data')
        if use_types:
            men_embeddings, _, men_idxs_by_type = data_process.embed_and_index(
                bi_reranker, men_vecs, encoder_type="context", alpha=params["alpha"], n_gpu=n_gpu, corpus=processed_data,
                force_exact_search=True, batch_size=params['embed_batch_size'])
        else:
            men_embeddings = data_process.embed_and_index(bi_reranker, men_vecs,
                                                                encoder_type="context", 
                                                                alpha=params["alpha"],
                                                                n_gpu=n_gpu,
                                                                force_exact_search=True,
                                                                batch_size=params['embed_batch_size'],
                                                                only_embed=True)
        logger.info('Biencoder: Embedding finished')

        logger.info("Biencoder: Finding nearest entities for each mention...")
        if not use_types:
            _, bi_dict_nns = dict_index.search(men_embeddings, topk)
        else:
            bi_dict_nns = np.zeros((len(men_embeddings), topk), dtype=int)
            for entity_type in men_idxs_by_type:
                men_embeds_by_type = men_embeddings[men_idxs_by_type[entity_type]]
                _, dict_nns_by_type = dict_indexes[entity_type].search(men_embeds_by_type, topk)
                dict_nns_idxs = np.array(list(map(lambda x: dict_idxs_by_type[entity_type][x], dict_nns_by_type)))
                for i, idx in enumerate(men_idxs_by_type[entity_type]):
                    bi_dict_nns[idx] = dict_nns_idxs[i]
        logger.info("Biencoder: Search finished")

        labels = [-1]*len(bi_dict_nns)
        for men_idx in range(len(bi_dict_nns)):
            gold_idx = processed_data[men_idx]["label_idxs"][0]
            for i in range(len(bi_dict_nns[men_idx])):
                if bi_dict_nns[men_idx][i] == gold_idx:
                    labels[men_idx] = i
                    break
            
        titles = []
        for men_idx in range(len(bi_dict_nns)):
            title = [entity_dictionary[i]['title'] for i in bi_dict_nns[men_idx]]
            titles.append(title)

        logger.info(f"Biencoder: Saving top-{topk} biencoder candidates for {mode} set")
        save_data_path = os.path.join(params['output_path'], f'candidates_{mode}_top{topk}.t7')
        torch.save({
            "mode": mode,
            "candidates": bi_dict_nns,
            "labels": labels,
            "titles": titles
        }, save_data_path)
        logger.info("Biencoder: Saved")

def main(params):
    # Parameter initializations
    logger = utils.get_logger(params["output_path"])
    global SCORING_BATCH_SIZE
    SCORING_BATCH_SIZE = params["scoring_batch_size"]
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pickle_src_path = params["pickle_src_path"]
    if pickle_src_path is None or not os.path.exists(pickle_src_path):
        pickle_src_path = output_path
    biencoder_indices_path = params["biencoder_indices_path"]
    if biencoder_indices_path is None:
        biencoder_indices_path = output_path
    elif not os.path.exists(biencoder_indices_path):
        os.makedirs(biencoder_indices_path)
    max_k = params["knn"]  # Maximum k-NN graph to build for evaluation
    use_types = params["use_types"]
    within_doc = params["within_doc"]
    discovery_mode = params["discovery"]

    # Bi-encoder model
    biencoder_params = copy.deepcopy(params)
    biencoder_params['add_linear'] = False
    bi_reranker = BiEncoderRanker(biencoder_params)
    bi_tokenizer = bi_reranker.tokenizer
    k_biencoder = params["bi_knn"]  # Number of biencoder nearest-neighbors to fetch for cross-encoder scoring (default: 64)

    # Cross-encoder model
    params['add_linear'] = True
    params['add_sigmoid'] = True
    cross_reranker = CrossEncoderRanker(params)
    n_gpu = cross_reranker.n_gpu
    cross_reranker.model.eval()

    # Input lengths
    max_seq_length = params["max_seq_length"]
    max_context_length = params["max_context_length"]
    max_cand_length = params["max_cand_length"]

    # Fix random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cross_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # The below code is to generate the candidates for cross-encoder training and inference

    save_topk_biencoder_cands(bi_reranker,
                                use_types, logger, n_gpu, params,
                                bi_tokenizer, max_context_length,
                                max_cand_length, pickle_src_path, topk=64)
    exit()


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    parser.add_argument(
        "--path_to_gold",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1
    )

    parser.add_argument(
        "--path_to_prediction",
        type=str,
        default=None
    )
    parser.add_argument(
        "--pred_json",
        type=str,
        default=None
    )

    parser.add_joint_train_args()
    args = parser.parse_args()
    print(args)
    main(args.__dict__)