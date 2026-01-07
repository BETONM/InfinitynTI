# Textual Inversion (TI) 학습 및 Inference 명령어 가이드

## 개요
이 문서는 Infinity 모델에 Textual Inversion을 적용하여 학습하고, 학습된 임베딩을 사용하여 이미지를 생성하는 방법을 설명합니다.

**학습 프롬프트 형식**: `"A photo of a {initializer_token}"` (예: `"A photo of a dog"`)  
**추론 프롬프트 형식**: `"A photo of a {learned_token}"` (예: `"A photo of a dog2"`)

---

## 1. TI 학습 (Training)

### 1.1 데이터 준비
```bash
# TI 학습용 데이터셋 경로
# data/TI/ 디렉토리에 jsonl 파일과 이미지가 있어야 함
data_path='data/TI'
```

### 1.2 단일 GPU 학습 (로컬 테스트)
```bash
cd Infinity

# 환경 변수 설정
export SINGLE=1
export OMP_NUM_THREADS=8

# TI 학습 실행
torchrun \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=12345 \
train.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path local_output/ti_dog \
--task_type='t2i' \
--bed=checkpoints/ti_dog/ \
--data_path=data/TI \
--exp_name=ti_dog \
--tblr=6e-3 \
--pn 0.06M \
--model=2bc8 \
--lbs=4 \
--workers=8 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path=weights/flan-t5-xl \
--vae_type 32 \
--vae_ckpt=weights/infinity_vae_d32_rdn_short.pth \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 100 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--apply_spatial_patchify 0 \
--use_flex_attn=True \
--pad=128 \
--ti_enable=1 \
--ti_initializer_token="dog" \
--ti_lr=1e-4 \
--ti_embed_path="ti_embedding_dog2.pt"
```

### 1.3 멀티 GPU 학습 (분산 학습)
```bash
cd Infinity

# 환경 변수 설정 (분산 학습 환경에 맞게 수정)
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

# 분산 학습 실행
torchrun \
--nproc_per_node=${ARNOLD_WORKER_GPU} \
--nnodes=${ARNOLD_WORKER_NUM} \
--node_rank=${ARNOLD_ID} \
--master_addr=${METIS_WORKER_0_HOST} \
--master_port=${METIS_WORKER_0_PORT} \
train.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path local_output/ti_dog \
--task_type='t2i' \
--bed=checkpoints/ti_dog/ \
--data_path=data/TI \
--exp_name=ti_dog \
--tblr=6e-3 \
--pn 0.06M \
--model=2bc8 \
--lbs=4 \
--workers=8 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path=weights/flan-t5-xl \
--vae_type 32 \
--vae_ckpt=weights/infinity_vae_d32_rdn_short.pth \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 100 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--apply_spatial_patchify 0 \
--use_flex_attn=True \
--pad=128 \
--ti_enable=1 \
--ti_initializer_token="dog" \
--ti_lr=1e-4 \
--ti_embed_path="ti_embedding_dog2.pt"
```

### 1.4 TI 학습 관련 주요 인자 설명

| 인자 | 설명 | 기본값 | 예시 |
|------|------|-------|------|
| `--ti_enable` | TI 학습 활성화 | `0` (False) | `1` |
| `--ti_initializer_token` | 초기화에 사용할 토큰 | `"dog"` | `"dog"`, `"cat"` |
| `--ti_lr` | TI 임베딩 학습률 | `1e-4` | `1e-4`, `5e-5` |
| `--ti_embed_path` | 학습된 TI 임베딩 저장 경로 | `"ti_embedding_dog2.pt"` | `"ti_embedding_dog2.pt"` |
| `--data_path` | TI 학습 데이터셋 경로 | - | `"data/TI"` |

**중요**: 
- `--ti_enable=1`을 설정하면 GPT 모델 파라미터는 자동으로 freeze되고, TI 임베딩만 학습됩니다.
- 학습이 완료되면 `--ti_embed_path`에 지정한 경로에 TI 임베딩이 저장됩니다.

---

## 2. TI Inference (이미지 생성)

### 2.1 기본 Inference (TI 임베딩 사용)
```bash
cd Infinity

# 환경 변수 설정
export TI_EMBED_PATH="ti_embedding_dog2.pt"
export TI_LEARNED_TOKEN="dog2"

# Inference 실행
python3 tools/run_infinity.py \
--cfg 4 \
--tau 0.5 \
--pn 1M \
--model_path weights/infinity_2b_reg.pth \
--vae_type 32 \
--vae_path weights/infinity_vae_d32_reg.pth \
--add_lvl_embeding_only_first_block 1 \
--use_bit_label 1 \
--model_type infinity_2b \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_scale_schedule_embedding 0 \
--checkpoint_type torch \
--text_encoder_ckpt weights/flan-t5-xl \
--text_channels 2048 \
--apply_spatial_patchify 0 \
--prompt "A photo of a dog2" \
--seed 1 \
--save_file output_dog2.jpg
```

### 2.2 환경 변수 없이 직접 지정
```bash
cd Infinity

# TI 임베딩 경로와 learned token을 환경 변수로 설정
TI_EMBED_PATH="ti_embedding_dog2.pt" \
TI_LEARNED_TOKEN="dog2" \
python3 tools/run_infinity.py \
--cfg 4 \
--tau 0.5 \
--pn 1M \
--model_path weights/infinity_2b_reg.pth \
--vae_type 32 \
--vae_path weights/infinity_vae_d32_reg.pth \
--add_lvl_embeding_only_first_block 1 \
--use_bit_label 1 \
--model_type infinity_2b \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_scale_schedule_embedding 0 \
--checkpoint_type torch \
--text_encoder_ckpt weights/flan-t5-xl \
--text_channels 2048 \
--apply_spatial_patchify 0 \
--prompt "A photo of a dog2" \
--seed 1 \
--save_file output_dog2.jpg
```

### 2.3 다양한 프롬프트 예시
```bash
# 기본 사용
--prompt "A photo of a dog2"

# 더 복잡한 프롬프트
--prompt "A photo of a dog2 playing in the park"

# 스타일 적용
--prompt "A photo of a dog2, oil painting style"

# 배경 추가
--prompt "A photo of a dog2 on a beach at sunset"
```

### 2.4 Inference 관련 환경 변수

| 환경 변수 | 설명 | 기본값 | 예시 |
|-----------|------|-------|------|
| `TI_EMBED_PATH` | 학습된 TI 임베딩 파일 경로 | `"ti_embedding_dog2.pt"` | `"ti_embedding_dog2.pt"` |
| `TI_LEARNED_TOKEN` | 프롬프트에서 사용할 learned token | `"dog2"` | `"dog2"`, `"cat2"` |

**중요**: 
- 프롬프트에 `TI_LEARNED_TOKEN` (예: `"dog2"`)이 포함되어 있어야 TI 임베딩이 주입됩니다.
- 프롬프트 형식은 `"A photo of a {learned_token}"`이어야 합니다.

---

## 3. 전체 워크플로우 예시

### Step 1: 데이터 준비
```bash
# 1. TI 학습용 이미지 준비 (dog 사진 5개)
mkdir -p Infinity/data/TI/images
# dog_001.jpg, dog_002.jpg, ..., dog_005.jpg를 data/TI/images/에 복사

# 2. jsonl 파일 확인 (이미 생성됨)
cat Infinity/data/TI/1.000_0000000005.jsonl
```

### Step 2: TI 학습
```bash
cd Infinity

torchrun \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=12345 \
train.py \
--ep=100 \
--data_path=data/TI \
--ti_enable=1 \
--ti_initializer_token="dog" \
--ti_lr=1e-4 \
--ti_embed_path="ti_embedding_dog2.pt" \
# ... (기타 인자들)
```

### Step 3: 학습 완료 확인
```bash
# TI 임베딩 파일 확인
ls -lh ti_embedding_dog2.pt

# 학습 로그 확인
tail -f local_output/ti_dog/log.txt
```

### Step 4: Inference
```bash
cd Infinity

export TI_EMBED_PATH="ti_embedding_dog2.pt"
export TI_LEARNED_TOKEN="dog2"

python3 tools/run_infinity.py \
--prompt "A photo of a dog2" \
--save_file output_dog2.jpg \
# ... (기타 인자들)
```

---

## 4. 주의사항

### 학습 시
1. **데이터 형식**: 모든 학습 데이터의 프롬프트는 `"A photo of a {initializer_token}"` 형식이어야 합니다.
2. **학습률**: TI 임베딩은 작은 학습률(`1e-4` ~ `5e-5`)을 사용하는 것이 좋습니다.
3. **GPU 메모리**: GPT 모델 파라미터가 freeze되므로 메모리 사용량이 줄어듭니다.
4. **체크포인트**: 학습된 TI 임베딩은 체크포인트와 별도로 `--ti_embed_path`에 저장됩니다.

### Inference 시
1. **프롬프트 형식**: 반드시 `"A photo of a {learned_token}"` 형식을 사용해야 합니다.
2. **환경 변수**: `TI_EMBED_PATH`와 `TI_LEARNED_TOKEN`이 올바르게 설정되어 있는지 확인하세요.
3. **파일 경로**: `TI_EMBED_PATH`에 지정한 파일이 존재하는지 확인하세요.

---

## 5. 트러블슈팅

### 문제: TI 임베딩이 주입되지 않음
- **원인**: 프롬프트에 `TI_LEARNED_TOKEN`이 없거나, 형식이 맞지 않음
- **해결**: 프롬프트를 `"A photo of a dog2"` 형식으로 수정

### 문제: 학습 중 에러 발생
- **원인**: `--ti_enable=1`이지만 `--online_t5=1`이 설정되지 않음
- **해결**: `--online_t5=1` 추가

### 문제: Inference 시 TI 임베딩 파일을 찾을 수 없음
- **원인**: `TI_EMBED_PATH` 경로가 잘못됨
- **해결**: 절대 경로 사용 또는 현재 디렉토리에서 상대 경로 확인

---

## 6. 빠른 참조

### 학습 (최소 명령어)
```bash
torchrun --nproc_per_node=1 train.py \
--data_path=data/TI \
--ti_enable=1 \
--ti_initializer_token="dog" \
--ti_embed_path="ti_embedding_dog2.pt" \
# ... (기타 필수 인자)
```

### Inference (최소 명령어)
```bash
TI_EMBED_PATH="ti_embedding_dog2.pt" TI_LEARNED_TOKEN="dog2" \
python3 tools/run_infinity.py \
--prompt "A photo of a dog2" \
--save_file output.jpg \
# ... (기타 필수 인자)
```

