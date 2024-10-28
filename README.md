# DynamicER

This is the official repository of our **EMNLP 2024** paper:\
**[DynamicER: Resolving Emerging Mentions to Dynamic Entities for RAG](https://arxiv.org/abs/2410.11494)**

<img src="asset/figure1.png" alt="Figure 1" width="500"/>

If you find our resources useful, please cite our work:

```bib
@inproceedings{kim-etal-2024-dynamicer,
  title="DynamicER: Resolving Emerging Mentions to Dynamic Entities for RAG",
  author="Kim, Jinyoung and Ko, Dayoon and Kim, Gunhee",
  booktitle="Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
  year="2024"
}
```

## Dataset

### Note for Dataset Users

- Usage of DynamicER is subject to [Tumblr Terms of Service](https://www.tumblr.com/policy/en/terms-of-service) and [User Guidelines](https://www.tumblr.com/policy/en/user-guidelines). Users must agree to and comply with the [Tumblr Application Developer and API License Agreement](https://www.tumblr.com/docs/en/api_agreement).

- DynamicER is intended solely for non-commercial research purposes. Commercial or for-profit uses are restricted; DynamicER must not be used to train models for deployment in production systems associated with business or government agency products. For more details, please refer to the Ethics Statement in the paper.

### Dataset File
You can download our dataset file by clicking the link below:
- Google Drive Link : [Download](https://drive.google.com/file/d/1qg_4IHdKjSb3JJla7AFD7DiomCuuMHLy/view?usp=sharing)

### Dataset Configuration
Please check [DATA_CONFIG.md](DATA_CONFIG.md) for configuration details of the dataset files.


## TempCCA

### Environment Setup

We provide a conda environment for setup. The following commands will create a new conda environment with all dependencies:

```
conda env create -f environment.yaml
conda activate tempcca
```

### Training and Inference

For continuous **training** (for each timestep 230820, 231020), use the following command:
```
PYTHONPATH='.' python blink/biencoder/proposed_train.py --bert_model=bert-base-cased --data_path=data/tempcca/processed/{timestep} --output_path=models/trained/tempcca_mst/{timestep} --pickle_src_path=models/trained/tempcca/{timestep} --path_to_model=models/trained/tempcca_mst/{prev_timestep}/epoch_4/pytorch_model.bin --num_train_epochs=5 --train_batch_size=32 --gradient_accumulation_steps=4 --eval_interval=10000 --pos_neg_loss --force_exact_search --embed_batch_size=1000 --data_parallel --path_to_prev_gold=models/trained/tempcca_mst/{prev_timestep} --alpha=0.8
```

For continuous **inference** (for each timestep 231220, 240220, 240420), use the following command:
```
PYTHONPATH='.' python blink/biencoder/eval_cluster_linking.py --bert_model=bert-base-cased --data_path=data/tempcca/processed/{timestep} --output_path=models/trained/tempcca_mst/eval/{timestep} --pickle_src_path=models/trained/tempcca/eval/{timestep} --path_to_model=models/trained/tempcca_mst/231020/epoch_4/pytorch_model.bin --recall_k=64 --embed_batch_size=1000 --force_exact_search --data_parallel --path_to_prev_gold=models/trained/tempcca_mst/231020 --alpha=0.8
```

For **entity resolution in entity-centric QA** (for each timestep 231220, 240220, 240420), use the following command:
```
PYTHONPATH='.' python blink/crossencoder/qa_inference.py --data_path=data/tempcca/processed/{timestep} --output_path=models/trained/tempcca/{timestep}/qa_prediction/pred --pickle_src_path=models/trained/tempcca/{timestep}/qa_prediction --path_to_biencoder_model=models/trained/tempcca_mst/231020/epoch_4/pytorch_model.bin --bert_model=bert-base-cased --data_parallel --scoring_batch_size=64 --save_topk_result --path_to_prediction=models/trained/tempcca_mst/eval/231220 --pred_json=eval_results_{unix_timestamp}-directed-0.json --alpha=0.8
```

## Acknowledgements

We acknowledge the use of skeleton code from [BLINK](https://github.com/facebookresearch/BLINK) and [ArboEL](https://github.com/dhdhagar/arboEL), which provided the base infrastructure for this project.