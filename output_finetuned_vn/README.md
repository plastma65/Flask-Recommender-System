---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:158
- loss:CosineSimilarityLoss
base_model: vinai/phobert-base
widget:
- source_sentence: Thực hành CTDL & thuật giải | 6 | Biểu diễn cấu trúc dữ liệu, áp
    dụng thuật giải xử lý dữ liệu. | HK04
  sentences:
  - Thực hành hệ quản trị cơ sở dữ liệu | 6 | Cài đặt, quản trị và vận hành hệ quản
    trị cơ sở dữ liệu; bảo mật, sao lưu, phục hồi. | HK06
  - 'Thực hành Cơ sở dữ liệu | 6 | Thực hành SQL Server: truy vấn SQL, xây dựng và
    điều khiển dữ liệu. | HK04'
  - Thực hành hệ quản trị cơ sở dữ liệu | 6 | Cài đặt, quản trị và vận hành hệ quản
    trị cơ sở dữ liệu; bảo mật, sao lưu, phục hồi. | HK06
- source_sentence: Thực hành LTHĐT | 6 | Hiện thực kỹ thuật OOP, xây dựng ứng dụng
    đơn giản với giao diện. | HK05
  sentences:
  - Thực hành hệ quản trị cơ sở dữ liệu | 6 | Cài đặt, quản trị và vận hành hệ quản
    trị cơ sở dữ liệu; bảo mật, sao lưu, phục hồi. | HK06
  - Thực hành LTHĐT | 6 | Hiện thực kỹ thuật OOP, xây dựng ứng dụng đơn giản với giao
    diện. | HK05
  - Chủ nghĩa xã hội khoa học | 6 | Quy luật khách quan của cách mạng XHCN; nội dung
    và quá trình phát triển CNXHKH. | HK03
- source_sentence: Thực hành LTHĐT | 6 | Hiện thực kỹ thuật OOP, xây dựng ứng dụng
    đơn giản với giao diện. | HK05
  sentences:
  - Thực hành Tin học đại cương | 6 | Thực hành Windows, Word, Excel, PowerPoint,
    Internet, email. | HK01
  - Mạng máy tính | 6 | Mô hình TCP/IP, định tuyến, DNS, email, FTP, VoIP. | HK05
  - Quản lý dự án | 6 | Khái niệm dự án, cấu trúc tổ chức, lập kế hoạch, tiến độ,
    kiểm soát; công cụ quản lý dự án, yếu tố ảnh hưởng đến thành công. | HK05
- source_sentence: Toán A1 (Hàm 1 biến, chuỗi) | 6 | Vi phân và tích phân hàm một
    biến; lý thuyết chuỗi; phương trình vi phân cơ bản. | HK01
  sentences:
  - Toán A1 (Hàm 1 biến, chuỗi) | 6 | Vi phân và tích phân hàm một biến; lý thuyết
    chuỗi; phương trình vi phân cơ bản. | HK01
  - 'Giáo dục thể chất 1 | 6 | Lý thuyết và thực hành TDTT: điền kinh, thể dục. |
    HK02'
  - 'Thực hành Lập trình ứng dụng cơ sở dữ liệu | 6 | Xây dựng ứng dụng Java: console,
    giao diện đồ họa, kết nối cơ sở dữ liệu. | HK06'
- source_sentence: Mạng máy tính | 6 | Mô hình TCP/IP, định tuyến, DNS, email, FTP,
    VoIP. | HK05
  sentences:
  - Mạng máy tính | 6 | Mô hình TCP/IP, định tuyến, DNS, email, FTP, VoIP. | HK05
  - 'Quản trị cơ sở dữ liệu | 6 | Bảo mật, phân quyền, sao lưu, phục hồi, tối ưu truy
    vấn; công cụ: SQL Server, Oracle. | HK06'
  - Triết học Mác - Lênin | 6 | Chủ nghĩa duy vật biện chứng, duy vật lịch sử; nhận
    thức luận, hình thái kinh tế – xã hội, giai cấp, nhà nước, cách mạng. | HK02
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on vinai/phobert-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [vinai/phobert-base](https://huggingface.co/vinai/phobert-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [vinai/phobert-base](https://huggingface.co/vinai/phobert-base) <!-- at revision c1e37c5c86f918761049cef6fa216b4779d0d01d -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Mạng máy tính | 6 | Mô hình TCP/IP, định tuyến, DNS, email, FTP, VoIP. | HK05',
    'Mạng máy tính | 6 | Mô hình TCP/IP, định tuyến, DNS, email, FTP, VoIP. | HK05',
    'Quản trị cơ sở dữ liệu | 6 | Bảo mật, phân quyền, sao lưu, phục hồi, tối ưu truy vấn; công cụ: SQL Server, Oracle. | HK06',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 158 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 158 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 23 tokens</li><li>mean: 38.06 tokens</li><li>max: 65 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 38.39 tokens</li><li>max: 65 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                            | sentence_1                                                                                                                                 | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Đồ án / Khóa luận tốt nghiệp | 6 | Nghiên cứu chuyên sâu: xây dựng hệ thống hoặc viết chuyên đề cuối khoá. | HK08</code>                                        | <code>Toán A3 (Đại số tuyến tính) | 6 | Ma trận, hệ phương trình tuyến tính, định thức, không gian vectơ, ánh xạ tuyến tính. | HK03</code> | <code>0.0</code> |
  | <code>Tư tưởng Hồ Chí Minh | 6 | Tư tưởng về độc lập dân tộc, CNXH, Đảng, nhà nước, văn hóa, đạo đức, con người. | HK05</code>                                        | <code>Tư tưởng Hồ Chí Minh | 6 | Tư tưởng về độc lập dân tộc, CNXH, Đảng, nhà nước, văn hóa, đạo đức, con người. | HK05</code>             | <code>1.0</code> |
  | <code>Tiếng Anh 1 | 6 | Trình độ sơ cấp: nghe, nói, đọc, viết; chủ đề: con người, nơi chốn, đồ vật, số đếm, quốc gia, thời gian rảnh, thức ăn, tiền tệ. | HK01</code> | <code>Lập trình Windows | 6 | Phát triển ứng dụng desktop với Windows Forms bằng C#. | HK07</code>                                         | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 4.1.0
- Transformers: 4.52.4
- PyTorch: 2.6.0+cpu
- Accelerate: 1.9.0
- Datasets: 4.0.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->