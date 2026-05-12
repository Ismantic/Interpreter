---
dataset_info:
- config_name: cs-en
  features:
  - name: translation
    struct:
    - name: Delta
      dtype: float64
    - name: alma_cs
      dtype: string
    - name: alma_cs_kiwi
      dtype: float64
    - name: alma_cs_kiwi_xcomet
      dtype: float64
    - name: alma_cs_xcomet
      dtype: float64
    - name: alma_en
      dtype: string
    - name: alma_en_kiwi
      dtype: float64
    - name: alma_en_kiwi_xcomet
      dtype: float64
    - name: alma_en_xcomet
      dtype: float64
    - name: cs
      dtype: string
    - name: en
      dtype: string
    - name: gpt4_cs
      dtype: string
    - name: gpt4_cs_kiwi
      dtype: float64
    - name: gpt4_cs_kiwi_xcomet
      dtype: float64
    - name: gpt4_cs_xcomet
      dtype: float64
    - name: gpt4_en
      dtype: string
    - name: gpt4_en_kiwi
      dtype: float64
    - name: gpt4_en_kiwi_xcomet
      dtype: float64
    - name: gpt4_en_xcomet
      dtype: float64
    - name: language_pair
      dtype: string
    - name: ref_cs_kiwi
      dtype: float64
    - name: ref_cs_kiwi_xcomet
      dtype: float64
    - name: ref_cs_xcomet
      dtype: float64
    - name: ref_en_kiwi
      dtype: float64
    - name: ref_en_kiwi_xcomet
      dtype: float64
    - name: ref_en_xcomet
      dtype: float64
    - name: required_directions
      dtype: string
  splits:
  - name: train
    num_bytes: 1973638
    num_examples: 2009
  download_size: 1407107
  dataset_size: 1973638
- config_name: de-en
  features:
  - name: translation
    struct:
    - name: Delta
      dtype: float64
    - name: alma_de
      dtype: string
    - name: alma_de_kiwi
      dtype: float64
    - name: alma_de_kiwi_xcomet
      dtype: float64
    - name: alma_de_xcomet
      dtype: float64
    - name: alma_en
      dtype: string
    - name: alma_en_kiwi
      dtype: float64
    - name: alma_en_kiwi_xcomet
      dtype: float64
    - name: alma_en_xcomet
      dtype: float64
    - name: de
      dtype: string
    - name: en
      dtype: string
    - name: gpt4_de
      dtype: string
    - name: gpt4_de_kiwi
      dtype: float64
    - name: gpt4_de_kiwi_xcomet
      dtype: float64
    - name: gpt4_de_xcomet
      dtype: float64
    - name: gpt4_en
      dtype: string
    - name: gpt4_en_kiwi
      dtype: float64
    - name: gpt4_en_kiwi_xcomet
      dtype: float64
    - name: gpt4_en_xcomet
      dtype: float64
    - name: language_pair
      dtype: string
    - name: ref_de_kiwi
      dtype: float64
    - name: ref_de_kiwi_xcomet
      dtype: float64
    - name: ref_de_xcomet
      dtype: float64
    - name: ref_en_kiwi
      dtype: float64
    - name: ref_en_kiwi_xcomet
      dtype: float64
    - name: ref_en_xcomet
      dtype: float64
    - name: required_directions
      dtype: string
  splits:
  - name: train
    num_bytes: 2743275
    num_examples: 3065
  download_size: 1782879
  dataset_size: 2743275
- config_name: is-en
  features:
  - name: translation
    struct:
    - name: Delta
      dtype: float64
    - name: alma_en
      dtype: string
    - name: alma_en_kiwi
      dtype: float64
    - name: alma_en_kiwi_xcomet
      dtype: float64
    - name: alma_en_xcomet
      dtype: float64
    - name: alma_is
      dtype: string
    - name: alma_is_kiwi
      dtype: float64
    - name: alma_is_kiwi_xcomet
      dtype: float64
    - name: alma_is_xcomet
      dtype: float64
    - name: en
      dtype: string
    - name: gpt4_en
      dtype: string
    - name: gpt4_en_kiwi
      dtype: float64
    - name: gpt4_en_kiwi_xcomet
      dtype: float64
    - name: gpt4_en_xcomet
      dtype: float64
    - name: gpt4_is
      dtype: string
    - name: gpt4_is_kiwi
      dtype: float64
    - name: gpt4_is_kiwi_xcomet
      dtype: float64
    - name: gpt4_is_xcomet
      dtype: float64
    - name: is
      dtype: string
    - name: language_pair
      dtype: string
    - name: ref_en_kiwi
      dtype: float64
    - name: ref_en_kiwi_xcomet
      dtype: float64
    - name: ref_en_xcomet
      dtype: float64
    - name: ref_is_kiwi
      dtype: float64
    - name: ref_is_kiwi_xcomet
      dtype: float64
    - name: ref_is_xcomet
      dtype: float64
    - name: required_directions
      dtype: string
  splits:
  - name: train
    num_bytes: 1990606
    num_examples: 2009
  download_size: 1385693
  dataset_size: 1990606
- config_name: ru-en
  features:
  - name: translation
    struct:
    - name: Delta
      dtype: float64
    - name: alma_en
      dtype: string
    - name: alma_en_kiwi
      dtype: float64
    - name: alma_en_kiwi_xcomet
      dtype: float64
    - name: alma_en_xcomet
      dtype: float64
    - name: alma_ru
      dtype: string
    - name: alma_ru_kiwi
      dtype: float64
    - name: alma_ru_kiwi_xcomet
      dtype: float64
    - name: alma_ru_xcomet
      dtype: float64
    - name: en
      dtype: string
    - name: gpt4_en
      dtype: string
    - name: gpt4_en_kiwi
      dtype: float64
    - name: gpt4_en_kiwi_xcomet
      dtype: float64
    - name: gpt4_en_xcomet
      dtype: float64
    - name: gpt4_ru
      dtype: string
    - name: gpt4_ru_kiwi
      dtype: float64
    - name: gpt4_ru_kiwi_xcomet
      dtype: float64
    - name: gpt4_ru_xcomet
      dtype: float64
    - name: language_pair
      dtype: string
    - name: ref_en_kiwi
      dtype: float64
    - name: ref_en_kiwi_xcomet
      dtype: float64
    - name: ref_en_xcomet
      dtype: float64
    - name: ref_ru_kiwi
      dtype: float64
    - name: ref_ru_kiwi_xcomet
      dtype: float64
    - name: ref_ru_xcomet
      dtype: float64
    - name: required_directions
      dtype: string
    - name: ru
      dtype: string
  splits:
  - name: train
    num_bytes: 2666563
    num_examples: 2009
  download_size: 1627361
  dataset_size: 2666563
- config_name: zh-en
  features:
  - name: translation
    struct:
    - name: Delta
      dtype: float64
    - name: alma_en
      dtype: string
    - name: alma_en_kiwi
      dtype: float64
    - name: alma_en_kiwi_xcomet
      dtype: float64
    - name: alma_en_xcomet
      dtype: float64
    - name: alma_zh
      dtype: string
    - name: alma_zh_kiwi
      dtype: float64
    - name: alma_zh_kiwi_xcomet
      dtype: float64
    - name: alma_zh_xcomet
      dtype: float64
    - name: en
      dtype: string
    - name: gpt4_en
      dtype: string
    - name: gpt4_en_kiwi
      dtype: float64
    - name: gpt4_en_kiwi_xcomet
      dtype: float64
    - name: gpt4_en_xcomet
      dtype: float64
    - name: gpt4_zh
      dtype: string
    - name: gpt4_zh_kiwi
      dtype: float64
    - name: gpt4_zh_kiwi_xcomet
      dtype: float64
    - name: gpt4_zh_xcomet
      dtype: float64
    - name: language_pair
      dtype: string
    - name: ref_en_kiwi
      dtype: float64
    - name: ref_en_kiwi_xcomet
      dtype: float64
    - name: ref_en_xcomet
      dtype: float64
    - name: ref_zh_kiwi
      dtype: float64
    - name: ref_zh_kiwi_xcomet
      dtype: float64
    - name: ref_zh_xcomet
      dtype: float64
    - name: required_directions
      dtype: string
    - name: zh
      dtype: string
  splits:
  - name: train
    num_bytes: 2462110
    num_examples: 3065
  download_size: 1697255
  dataset_size: 2462110
configs:
- config_name: cs-en
  data_files:
  - split: train
    path: cs-en/train-*
- config_name: de-en
  data_files:
  - split: train
    path: de-en/train-*
- config_name: is-en
  data_files:
  - split: train
    path: is-en/train-*
- config_name: ru-en
  data_files:
  - split: train
    path: ru-en/train-*
- config_name: zh-en
  data_files:
  - split: train
    path: zh-en/train-*
license: mit
task_categories:
- translation
language:
- ru
- cs
- zh
- is
- de
---
# Dataset Card for "ALMA-R-Preference"

This is triplet preference data used by [ALMA-R](https://arxiv.org/abs/2401.08417) model.

The triplet preference data, supporting 10 translation directions, is built upon the FLORES-200 development and test data. For each direction, we provide a source sentence along with three translations: one from GPT-4, another from ALMA-13B-LoRA, and a reference translation. For instance, in the English-German pair, our data structure is as follows:

### Sentences:
- de: Original German sentence
- en: Original English sentence
- alma_de: German sentence translated from English by ALMA
- gpt4_de: German sentence translated from English by GPT-4
- alma_en: English sentence translated from German by ALMA
- gpt4_en: English sentence translated from German by GPT-4

### Scores
- alma_en_${Score}: ${Score} of English sentence translated by ALMA
- gpt4_en_${Score}: ${Score} of English sentence translated by GPT4
- ref_en_${Score}: ${Score} of reference English sentence
- alma_de_${Score}: ${Score}  of German sentence translated by ALMA
- gpt4_de_${Sscore}: ${Score} of German sentence translated by GPT4
- ref_en_${Score}: ${Score} of reference German sentence

${Score} can be numbers from kiwi ([wmt23-cometkiwi-da-xxl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl)), xcomet ([XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL)), 
or kiwi_xcomet (average score of kiwi and xcomet).

### Others
- Delta: A value of 0 indicates non-human annotated data or tied evaluations. A postive number suggests that gpt4_de is better than alma_de, vice versa
- required_directions: An empty field implies that this data point can be used for both translation directions. If the string 'en-de' is specified, it indicates that this data point is exclusively for English to German translation

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