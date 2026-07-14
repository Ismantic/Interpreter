---
dataset_info:
- config_name: cs-en
  features:
  - name: translation
    struct:
    - name: cs
      dtype: string
    - name: en
      dtype: string
  splits:
  - name: train
    num_bytes: 3432181
    num_examples: 12076
  - name: validation
    num_bytes: 318813
    num_examples: 1002
  download_size: 0
  dataset_size: 3750994
- config_name: de-en
  features:
  - name: translation
    struct:
    - name: de
      dtype: string
    - name: en
      dtype: string
  splits:
  - name: train
    num_bytes: 4108729
    num_examples: 14211
  - name: validation
    num_bytes: 329855
    num_examples: 1002
  download_size: 0
  dataset_size: 4438584
- config_name: is-en
  features:
  - name: translation
    struct:
    - name: is
      dtype: string
    - name: en
      dtype: string
  splits:
  - name: train
    num_bytes: 554190
    num_examples: 2009
  download_size: 0
  dataset_size: 554190
- config_name: ru-en
  features:
  - name: translation
    struct:
    - name: ru
      dtype: string
    - name: en
      dtype: string
  splits:
  - name: train
    num_bytes: 5427552
    num_examples: 15000
  - name: validation
    num_bytes: 442271
    num_examples: 1002
  download_size: 0
  dataset_size: 5869823
- config_name: zh-en
  features:
  - name: translation
    struct:
    - name: zh
      dtype: string
    - name: en
      dtype: string
  splits:
  - name: train
    num_bytes: 4700299
    num_examples: 15406
  - name: validation
    num_bytes: 285969
    num_examples: 1002
  download_size: 0
  dataset_size: 4986268
configs:
- config_name: cs-en
  data_files:
  - split: train
    path: cs-en/train-*
  - split: validation
    path: cs-en/validation-*
- config_name: de-en
  data_files:
  - split: train
    path: de-en/train-*
  - split: validation
    path: de-en/validation-*
- config_name: is-en
  data_files:
  - split: train
    path: is-en/train-*
- config_name: ru-en
  data_files:
  - split: train
    path: ru-en/train-*
  - split: validation
    path: ru-en/validation-*
- config_name: zh-en
  data_files:
  - split: train
    path: zh-en/train-*
  - split: validation
    path: zh-en/validation-*
---
# Dataset Card for "ALMA-Human-Parallel"

This is human-written parallel dataset used by [ALMA](https://arxiv.org/abs/2309.11674) translation models.

```
@misc{xu2023paradigm,
      title={A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models}, 
      author={Haoran Xu and Young Jin Kim and Amr Sharaf and Hany Hassan Awadalla},
      year={2023},
      eprint={2309.11674},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@misc{xu2024contrastive,
      title={Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation}, 
      author={Haoran Xu and Amr Sharaf and Yunmo Chen and Weiting Tan and Lingfeng Shen and Benjamin Van Durme and Kenton Murray and Young Jin Kim},
      year={2024},
      eprint={2401.08417},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```