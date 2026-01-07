# Textual Inversion (TI) 구현 가이드

## 개요
이 문서는 Infinity 프로젝트에 Textual Inversion 알고리즘을 적용하기 위한 변경 지점과 구현 방법을 설명합니다.

**중요**: 모든 학습 프롬프트는 **"A photo of a {token}"** 형식을 사용합니다.
- 학습 시: "A photo of a {initializer_token}" (예: "A photo of a dog")
- 추론 시: "A photo of a {learned_token}" (예: "A photo of a dog2")

## TI 알고리즘 요약

**학습 프롬프트 형식**: 모든 학습 데이터는 **"A photo of a {initializer_token}"** 형식을 사용
- 예: "A photo of a dog", "A photo of a cat" 등
- initializer_token: 초기화에 사용할 토큰 (예: "dog", "cat")

1. 데이터 로드: 이미지와 텍스트 프롬프트 배치 로드 (프롬프트 형식: "A photo of a {token}")
2. 토크나이징: `input_ids, attn_mask = tokenizer("A photo of a {token}")`
3. 텍스트 임베딩 생성: `E = text_encoder(input_ids, attn_mask).last_hidden_state`
4. TI 임베딩 준비: 
   - 초기화 프롬프트 "A photo of a {initializer_token}"에서 {initializer_token} 위치의 임베딩 추출
   - `ti_embeddings = nn.Parameter(init_token_embedding.clone())`, `ti_embeddings.requires_grad = True`
5. 임베딩 치환: "A photo of a" 다음의 {token} 위치를 찾아 `E[:, pos, :] = ti_embeddings`
6. 텍스트 조건 주입: 치환된 `E`를 이미지 생성 모델의 conditioning 입력으로 전달
7. 이미지 토큰 예측: AR/VAR Transformer로 다음 이미지 토큰 분포 예측
8. 손실 계산: 예측 이미지 토큰 vs GT 이미지 토큰
9. 역전파: `loss.backward()` (ti_embeddings만 업데이트, 나머지 freeze)
10. 옵티마이저 스텝: `optimizer_ti.step()`, `optimizer_ti.zero_grad()`

---

## 변경 지점 상세 분석

### 1. TI 파라미터 초기화 및 관리
**파일**: `train.py`  
**위치**: `build_model_optimizer` 함수 (line 89-250)

#### 변경 내용:
- TI 임베딩 파라미터를 생성하고 관리하는 로직 추가 필요
- 학습 프롬프트 형식: **"A photo of a {initializer_token}"** (예: "A photo of a dog", "A photo of a cat")
- 학습 대상 토큰(initializer_token)의 원본 임베딩을 추출하여 초기화하고 `requires_grad=True` 설정
- TI 전용 옵티마이저 생성

**구현 예시 위치** (line 235 이후):
```python
# [TI] TI 임베딩 파라미터 초기화
# 학습 프롬프트 형식: "A photo of a {initializer_token}"

# 1. 초기화용 프롬프트 생성
init_prompt = f"A photo of a {args.ti_initializer_token}"  # 예: "A photo of a dog"

# 2. 토크나이징
init_tokens = text_tokenizer(init_prompt, max_length=text_tokenizer.model_max_length, 
                            padding='max_length', truncation=True, return_tensors='pt')
init_input_ids = init_tokens.input_ids.to(args.device)
init_mask = init_tokens.attention_mask.to(args.device)

# 3. 텍스트 인코더로 임베딩 생성
with torch.no_grad():
    init_embeddings = text_encoder(input_ids=init_input_ids, attention_mask=init_mask)['last_hidden_state'].float()

# 4. initializer_token 위치 찾기
# T5는 서브워드 토크나이저이므로 여러 토큰으로 분할될 수 있음
# 예: "dog" -> ["dog"] 또는 ["do", "g"] 등
init_tokenizer_result = text_tokenizer(args.ti_initializer_token, add_special_tokens=False, return_tensors='pt')
init_token_ids = init_tokenizer_result.input_ids[0].tolist()  # [token_id1, token_id2, ...]

# "A photo of a" 다음의 토큰 위치 찾기
base_prompt = "A photo of a"
base_tokens = text_tokenizer(base_prompt, add_special_tokens=False, return_tensors='pt')
base_token_count = len(base_tokens.input_ids[0])

# initializer_token의 시작 위치 (base_token_count 이후)
token_positions = list(range(base_token_count, base_token_count + len(init_token_ids)))

# 5. 해당 위치의 임베딩 추출
# 여러 서브워드인 경우 평균 또는 첫 번째 토큰 사용
if len(token_positions) == 1:
    init_token_embedding = init_embeddings[0, token_positions[0], :]
else:
    # 여러 서브워드인 경우 평균 사용
    init_token_embedding = init_embeddings[0, token_positions, :].mean(dim=0)

# 6. TI 파라미터 생성
ti_embeddings = nn.Parameter(init_token_embedding.clone())
ti_embeddings.requires_grad = True

# 7. TI 전용 옵티마이저 생성
optimizer_ti = torch.optim.AdamW([ti_embeddings], lr=args.ti_lr)
```

---

### 2. 텍스트 임베딩 생성 및 치환 (학습)
**파일**: `train.py`  
**위치**: `train_one_ep` 함수 내부 (line 430-538)

#### 변경 내용:
- 텍스트 임베딩 생성 후, "A photo of a {token}" 형식에서 {token} 위치를 찾아 TI 임베딩으로 치환
- 토큰 위치 인덱스 찾기 및 치환 로직 추가
- 학습 프롬프트 형식: **"A photo of a {initializer_token}"**

**구현 예시 위치** (line 489 이후):
```python
# Line 509: text_features = text_encoder(...).last_hidden_state.float()
# [TI] 여기서 text_features 생성 후 TI 임베딩으로 치환

# 학습 프롬프트 형식: "A photo of a {initializer_token}"
# 각 배치 샘플의 caption이 이 형식이라고 가정

# 1. "A photo of a" 다음의 토큰 위치 찾기
base_prompt = "A photo of a"
base_tokens = text_tokenizer(base_prompt, add_special_tokens=False, return_tensors='pt')
base_token_count = len(base_tokens.input_ids[0])

# 2. initializer_token의 토큰 ID 찾기
init_tokenizer_result = text_tokenizer(args.ti_initializer_token, add_special_tokens=False, return_tensors='pt')
init_token_ids = init_tokenizer_result.input_ids[0].tolist()  # [token_id1, token_id2, ...]
init_token_count = len(init_token_ids)

# 3. 각 배치 샘플마다 해당 위치의 text_features를 TI 임베딩으로 치환
B = text_features.shape[0]
for batch_idx in range(B):
    # 각 샘플의 실제 길이 확인
    actual_len = lens[batch_idx]
    
    # "A photo of a" 다음의 위치 계산 (패딩 제외)
    token_start_pos = base_token_count
    token_end_pos = token_start_pos + init_token_count
    
    # 실제 길이 내에 있는지 확인
    if token_end_pos <= actual_len:
        # 여러 서브워드인 경우 모든 위치에 동일한 TI 임베딩 사용
        for pos in range(token_start_pos, token_end_pos):
            text_features[batch_idx, pos, :] = ti_embeddings

# 4. 치환된 text_features를 사용하여 kv_compact 생성 (기존 코드 그대로 사용)
```

**구체적 위치**:
- **Line 486-489**: 토크나이징 및 텍스트 임베딩 생성
  ```python
  tokens = text_tokenizer(...)  # Line 486
  input_ids = tokens.input_ids.cuda(...)  # Line 487
  mask = tokens.attention_mask.cuda(...)  # Line 488
  text_features = text_encoder(...).last_hidden_state.float()  # Line 489
  ```
  **→ Line 489 직후에 TI 임베딩 치환 로직 추가**

- **Line 495-498**: kv_compact 생성
  ```python
  kv_compact = []
  for len_i, feat_i in zip(lens, text_features.unbind(0)):
      kv_compact.append(feat_i[:len_i])
  kv_compact = torch.cat(kv_compact, dim=0)
  ```
  **→ 이 부분은 치환된 text_features를 사용하므로 변경 불필요**

---

### 3. 모델 Forward에서 텍스트 임베딩 처리
**파일**: `infinity/models/infinity.py`  
**위치**: `Infinity.forward` 메서드 (line 366-453)

#### 변경 내용:
- Forward 과정에서 이미 치환된 텍스트 임베딩을 사용하므로, 추가 치환은 불필요
- 단, 추론 시에는 학습된 TI 임베딩을 주입해야 함

**확인 위치**:
- **Line 382-394**: 텍스트 임베딩 처리
  ```python
  kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT  # Line 382
  # ... cond drop 처리 ...
  kv_compact = self.text_norm(kv_compact).contiguous()  # Line 390
  sos = cond_BD = self.text_proj_for_sos(...)  # Line 391
  kv_compact = self.text_proj_for_ca(kv_compact).contiguous()  # Line 392
  ```
  **→ 학습 시에는 train.py에서 이미 치환된 kv_compact가 들어오므로 변경 불필요**
  **→ 추론 시에는 별도 처리 필요 (아래 참조)**

---

### 4. 학습 루프에서 TI 파라미터 업데이트
**파일**: `trainer.py`  
**위치**: `InfinityTrainer.train_step` 메서드 (line 152-281)

#### 변경 내용:
- Forward pass는 그대로 사용
- Backward 시 TI 파라미터만 업데이트되도록 설정
- TI 전용 옵티마이저 스텝 추가

**구현 예시 위치** (line 217-235):
```python
# Line 218: grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(...)
# [TI] TI 파라미터만 업데이트
# TODO:
# 1. GPT 모델의 파라미터는 freeze (requires_grad=False 또는 gradient를 zero로)
# 2. optimizer_ti.step() 호출
# 3. optimizer_ti.zero_grad() 호출
```

**구체적 위치**:
- **Line 217-218**: Backward 및 gradient clipping
  ```python
  # [backward]
  grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(...)
  ```
  **→ 이 부분 이후에 TI 옵티마이저 스텝 추가**

- **Line 224-235**: Zero grad 및 EMA 업데이트
  ```python
  if stepping:
      if self.using_ema: self.ema_update(g_it)
      # ...
      self.gpt_opt.optimizer.zero_grad(set_to_none=True)
  ```
  **→ Line 235 직후에 `optimizer_ti.zero_grad()` 추가**

---

### 5. 추론 시 TI 임베딩 주입
**파일**: `infinity/models/infinity.py`  
**위치**: `Infinity.autoregressive_infer_cfg` 메서드 (line 455-642)

#### 변경 내용:
- 추론 시 학습된 TI 임베딩을 텍스트 임베딩에 주입
- 프롬프트 형식: **"A photo of a {learned_token}"** (예: "A photo of a dog2")

**구현 예시 위치** (line 484 이후):
```python
# Line 484: kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
# [TI] 추론 시 TI 임베딩 주입
# 프롬프트 형식: "A photo of a {learned_token}"

# 주의: label_B_or_BLT는 (kv_compact, lens, cu_seqlens_k, max_seqlen_k) 튜플이므로
# input_ids 정보가 별도로 필요할 수 있음
# 추론 인터페이스(run_infinity.py)에서 이미 주입된 상태로 전달되는 것이 더 효율적
```

**구체적 위치**:
- **Line 484**: 텍스트 조건 추출
  ```python
  kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
  ```
  **→ 이 부분 이후에 TI 임베딩 주입 로직 추가**

---

### 6. 추론 인터페이스에서 TI 임베딩 로드
**파일**: `tools/run_infinity.py`  
**위치**: `encode_prompt` 함수 (line 38-57)

#### 변경 내용:
- 추론 시 학습된 TI 임베딩을 로드하고 주입
- 프롬프트 형식: **"A photo of a {learned_token}"** (예: "A photo of a dog2")

**구현 예시 위치** (line 48 이후):
```python
# Line 48: text_features = text_encoder(...).last_hidden_state.float()
# [TI] 추론 시 TI 임베딩 주입
# 프롬프트 형식: "A photo of a {learned_token}"

# 1. 학습된 TI 임베딩 로드 (체크포인트에서)
# ti_embeddings = load_ti_embeddings_from_checkpoint(checkpoint_path)

# 2. 프롬프트에서 learned_token 찾기
# 예: prompt = "A photo of a dog2" -> learned_token = "dog2"
# 또는 args.ti_learned_token 사용

# 3. "A photo of a" 다음의 토큰 위치 찾기
base_prompt = "A photo of a"
base_tokens = text_tokenizer(base_prompt, add_special_tokens=False, return_tensors='pt')
base_token_count = len(base_tokens.input_ids[0])

# learned_token의 토큰 ID 찾기
learned_token_tokens = text_tokenizer(learned_token, add_special_tokens=False, return_tensors='pt')
learned_token_ids = learned_token_tokens.input_ids[0].tolist()
learned_token_count = len(learned_token_ids)

# 4. text_features에서 해당 위치에 TI 임베딩 주입
token_start_pos = base_token_count
token_end_pos = token_start_pos + learned_token_count
actual_len = lens[0]  # 배치 크기가 1이라고 가정

if token_end_pos <= actual_len:
    for pos in range(token_start_pos, token_end_pos):
        text_features[0, pos, :] = ti_embeddings
```

**구체적 위치**:
- **Line 45-48**: 토크나이징 및 텍스트 임베딩 생성
  ```python
  tokens = text_tokenizer(...)  # Line 45
  input_ids = tokens.input_ids.cuda(...)  # Line 46
  mask = tokens.attention_mask.cuda(...)  # Line 47
  text_features = text_encoder(...).last_hidden_state.float()  # Line 48
  ```
  **→ Line 48 직후에 TI 임베딩 주입 로직 추가**

---

### 7. TI 임베딩 저장/로드
**파일**: `trainer.py`  
**위치**: `InfinityTrainer.state_dict` 및 `load_state_dict` 메서드 (line 324-413)

#### 변경 내용:
- TI 임베딩 파라미터를 체크포인트에 저장/로드

**구현 예시 위치**:
- **Line 324-352**: `state_dict` 메서드
  ```python
  # [TI] TI 임베딩 저장
  # TODO: state['ti_embeddings'] = self.ti_embeddings.state_dict()
  ```

- **Line 354-413**: `load_state_dict` 메서드
  ```python
  # [TI] TI 임베딩 로드
  # TODO: if 'ti_embeddings' in state: self.ti_embeddings.load_state_dict(state['ti_embeddings'])
  ```

---

## 구현 순서

### Phase 1: 초기화 및 파라미터 관리
1. `train.py`의 `build_model_optimizer`에서 TI 파라미터 생성
2. TI 전용 옵티마이저 생성
3. `trainer.py`의 `__init__`에서 TI 관련 속성 초기화

### Phase 2: 학습 루프 수정
1. `train.py`의 `train_one_ep`에서 텍스트 임베딩 치환 로직 추가
2. `trainer.py`의 `train_step`에서 TI 옵티마이저 스텝 추가
3. GPT 모델 파라미터 freeze 처리

### Phase 3: 추론 수정
1. `infinity.py`의 `autoregressive_infer_cfg`에서 TI 임베딩 주입
2. `tools/run_infinity.py`의 `encode_prompt`에서 TI 임베딩 주입

### Phase 4: 저장/로드
1. `trainer.py`의 `state_dict`에서 TI 임베딩 저장
2. `trainer.py`의 `load_state_dict`에서 TI 임베딩 로드

---

## 주의사항

1. **프롬프트 형식**: 모든 학습 프롬프트는 **"A photo of a {initializer_token}"** 형식을 사용해야 함.
   - 예: "A photo of a dog", "A photo of a cat" 등
   - 이 형식을 통해 일관된 위치에서 TI 임베딩을 치환할 수 있음

2. **토큰 위치 찾기**: 
   - T5 토크나이저를 사용하므로, initializer_token이 여러 서브워드로 분할될 수 있음
   - 예: "dog" -> ["dog"] 또는 ["do", "g"] 등
   - "A photo of a" 다음의 모든 서브워드 위치를 찾아 치환해야 함
   - `base_token_count = len(text_tokenizer("A photo of a", ...).input_ids[0])`로 기준 위치 계산

3. **초기화**: 
   - TI 임베딩은 "A photo of a {initializer_token}" 프롬프트에서 initializer_token 위치의 원본 임베딩으로 초기화
   - 여러 서브워드인 경우 평균을 사용하거나 첫 번째 토큰 사용

4. **Gradient Flow**: GPT 모델의 파라미터는 freeze하되, 텍스트 인코더는 이미 freeze되어 있으므로 추가 작업 불필요.

5. **배치 처리**: 
   - 배치 내에서 각 샘플마다 동일한 형식("A photo of a {token}")을 사용하므로 위치 계산이 일관됨
   - 각 샘플의 실제 길이(lens)를 고려하여 마스킹 처리

6. **학습률**: TI 임베딩은 작은 학습률(예: 1e-4 ~ 1e-5)을 사용하는 것이 좋음.

7. **추론 시**: 
   - 추론 시에도 "A photo of a {learned_token}" 형식을 사용
   - learned_token은 학습된 TI 임베딩에 대응하는 토큰 (예: "dog2")

---

## 코드 변경 체크리스트

- [x] `train.py`: TI 파라미터 초기화 (build_model_optimizer) - **주석 추가 완료 (line 236)**
- [x] `train.py`: 텍스트 임베딩 치환 (train_one_ep, line 489 이후) - **주석 추가 완료 (line 490)**
- [x] `trainer.py`: TI 옵티마이저 초기화 (__init__) - **주석 추가 완료 (line 60)**
- [x] `trainer.py`: TI 옵티마이저 스텝 (train_step, line 222 이후) - **주석 추가 완료 (line 223, 239)**
- [x] `trainer.py`: TI 임베딩 저장/로드 (state_dict, load_state_dict) - **주석 추가 완료 (line 345, 377)**
- [x] `infinity.py`: 학습 시 텍스트 임베딩 처리 (forward, line 382) - **주석 추가 완료**
- [x] `infinity.py`: 추론 시 TI 임베딩 주입 (autoregressive_infer_cfg, line 484 이후) - **주석 추가 완료 (line 485)**
- [x] `tools/run_infinity.py`: 추론 인터페이스에서 TI 임베딩 주입 (encode_prompt, line 48 이후) - **주석 추가 완료 (line 49)**

**참고**: 모든 변경 지점에 `[TI]` 태그가 포함된 주석이 추가되었습니다. 실제 구현 시 해당 주석을 참고하여 코드를 작성하세요.

---

## 참고사항

- 현재 코드는 T5 텍스트 인코더를 사용하며, `text_encoder`는 이미 freeze되어 있음 (line 245-246 in train.py)
- 텍스트 임베딩은 `kv_compact` 형태로 처리되며, 각 샘플의 실제 길이만큼만 사용됨
- 학습 시에는 `train_one_ep`에서 텍스트 임베딩을 생성하고, `trainer.train_step`에서 모델 forward를 호출
- 추론 시에는 `autoregressive_infer_cfg`에서 텍스트 조건을 처리

