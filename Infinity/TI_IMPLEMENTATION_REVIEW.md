# TI 구현 로직 검토 결과

## 검토 요약

**현재 상태**: 모든 TI 관련 로직이 **주석(TODO)만 존재**하고, **실제 구현은 없음**

---

## 1. TI 임베딩 파라미터 로직 존재 여부

### ❌ **구현 없음**

**위치**: `train.py` line 238-256

**현재 상태**:
- 주석만 존재 (TODO)
- `ti_embeddings = nn.Parameter(...)` 생성 코드 없음
- `optimizer_ti` 생성 코드 없음

**필요한 구현**:
```python
# train.py의 build_model_optimizer 함수 내부 (line 238 이후)
# 1. TI 임베딩 파라미터 생성
ti_embeddings = nn.Parameter(init_token_embedding.clone())
ti_embeddings.requires_grad = True

# 2. TI 전용 옵티마이저 생성
optimizer_ti = torch.optim.AdamW([ti_embeddings], lr=args.ti_lr)

# 3. trainer에 전달 (trainer.__init__에서 받아야 함)
```

---

## 2. Optimizer가 s* (TI 임베딩)만 업데이트하는지

### ❌ **구현 없음**

**위치**: `trainer.py` line 224-231

**현재 상태**:
- 주석만 존재 (TODO)
- `self.optimizer_ti`가 존재하지 않음
- GPT 모델 파라미터 freeze 로직 없음

**문제점 분석**:

#### 2.1 현재 backward_clip_step 동작
```python
# trainer.py line 222
grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(...)
```

`backward_clip_step` 내부 동작 (`amp_opt.py` line 111-173):
1. `loss.backward()` 호출 → **모든 `requires_grad=True`인 파라미터에 대해 gradient 계산**
   - GPT 모델 파라미터
   - **TI 임베딩 파라미터 (만약 존재한다면)**
2. `self.optimizer.step()` 호출 → **GPT optimizer에 등록된 파라미터만 업데이트**
   - GPT 모델 파라미터만 업데이트
   - TI 파라미터는 optimizer에 등록되지 않았으므로 업데이트 안 됨

#### 2.2 현재 문제점

1. **GPT 모델 파라미터가 freeze되지 않음**
   - `requires_grad=False`로 설정하는 로직 없음
   - 불필요한 gradient 계산 발생 (메모리 낭비)

2. **TI optimizer가 존재하지 않음**
   - `self.optimizer_ti`가 없으므로 TI 파라미터 업데이트 불가

3. **Gradient가 누적됨**
   - `backward_clip_step`에서 `loss.backward()`가 모든 파라미터에 대해 gradient 계산
   - GPT optimizer는 GPT 파라미터만 업데이트하고 zero_grad
   - **TI 파라미터의 gradient는 남아있음** (별도 처리 필요)

**필요한 구현**:
```python
# trainer.py의 train_step 메서드 내부 (line 224 이후)

# 방법 1: GPT 모델 파라미터 freeze (권장)
# 학습 시작 전에 한 번만 실행
for param in self.gpt_wo_ddp.parameters():
    param.requires_grad = False

# 방법 2: backward 후 GPT 파라미터 gradient만 zero (매 step마다)
# backward_clip_step 이후
if stepping:
    # GPT optimizer는 이미 backward_clip_step에서 step() 호출됨
    # TI optimizer는 별도로 step() 호출
    if hasattr(self, 'optimizer_ti'):
        self.optimizer_ti.step()
        self.optimizer_ti.zero_grad()
    
    # GPT 파라미터의 gradient는 이미 zero_grad됨 (backward_clip_step 내부)
    # 하지만 명시적으로 확인
    self.gpt_opt.optimizer.zero_grad(set_to_none=True)
```

---

## 3. 전체 구현 상태 체크리스트

### ❌ 미구현 항목들

- [ ] **train.py**: TI 임베딩 파라미터 생성 (line 238)
- [ ] **train.py**: TI optimizer 생성 (line 238)
- [ ] **train.py**: trainer에 TI optimizer 전달
- [ ] **trainer.py**: TI optimizer 초기화 (line 60)
- [ ] **train.py**: 텍스트 임베딩 치환 로직 (line 511)
- [ ] **trainer.py**: GPT 모델 파라미터 freeze
- [ ] **trainer.py**: TI optimizer step (line 224)
- [ ] **trainer.py**: TI optimizer zero_grad (line 249)
- [ ] **trainer.py**: TI 임베딩 저장 (line 347)
- [ ] **trainer.py**: TI 임베딩 로드 (line 378)
- [ ] **infinity.py**: 추론 시 TI 임베딩 주입 (line 490)
- [ ] **run_infinity.py**: 추론 시 TI 임베딩 주입 (line 49)

---

## 4. 올바른 구현 흐름

### Phase 1: 초기화
```python
# train.py의 build_model_optimizer
# 1. TI 임베딩 파라미터 생성
ti_embeddings = nn.Parameter(...)
ti_embeddings.requires_grad = True

# 2. TI optimizer 생성
optimizer_ti = torch.optim.AdamW([ti_embeddings], lr=args.ti_lr)

# 3. GPT 모델 파라미터 freeze
for param in gpt_wo_ddp.parameters():
    param.requires_grad = False

# 4. trainer에 전달
trainer = InfinityTrainer(..., optimizer_ti=optimizer_ti, ti_embeddings=ti_embeddings)
```

### Phase 2: 학습 루프
```python
# train.py의 train_one_ep
# 1. 텍스트 임베딩 생성
text_features = text_encoder(...)

# 2. TI 임베딩으로 치환
text_features = replace_with_ti_embeddings(text_features, ti_embeddings, ...)

# trainer.py의 train_step
# 3. Forward & Loss 계산
loss = ...

# 4. Backward (모든 파라미터에 대해 gradient 계산)
#    - GPT 파라미터: requires_grad=False이므로 gradient 계산 안 됨
#    - TI 파라미터: requires_grad=True이므로 gradient 계산됨
loss.backward()

# 5. GPT optimizer step (실제로는 아무것도 업데이트 안 됨, requires_grad=False)
self.gpt_opt.backward_clip_step(...)

# 6. TI optimizer step (TI 파라미터만 업데이트)
self.optimizer_ti.step()
self.optimizer_ti.zero_grad()
```

---

## 5. 핵심 문제점

### 문제 1: GPT 모델 파라미터가 freeze되지 않음
- **현재**: GPT 파라미터가 `requires_grad=True` 상태
- **결과**: `loss.backward()` 시 GPT 파라미터에 대한 gradient도 계산됨 (불필요)
- **해결**: 학습 시작 전에 `param.requires_grad = False` 설정

### 문제 2: TI optimizer가 존재하지 않음
- **현재**: `self.optimizer_ti`가 없음
- **결과**: TI 파라미터를 업데이트할 수 없음
- **해결**: TI optimizer 생성 및 trainer에 전달

### 문제 3: Gradient 관리 문제
- **현재**: `backward_clip_step`이 GPT optimizer만 처리
- **결과**: TI 파라미터의 gradient가 남아있을 수 있음
- **해결**: TI optimizer의 `zero_grad()` 명시적 호출

---

## 6. 권장 구현 순서

1. **train.py**: TI 임베딩 파라미터 및 optimizer 생성
2. **train.py**: GPT 모델 파라미터 freeze
3. **trainer.py**: TI optimizer 및 임베딩 저장
4. **train.py**: 텍스트 임베딩 치환 로직
5. **trainer.py**: TI optimizer step 및 zero_grad
6. **trainer.py**: 저장/로드 로직
7. **추론 코드**: TI 임베딩 주입 로직

---

## 결론

**현재 제시된 로직에는 실제 구현이 전혀 없으며, 주석만 존재합니다.**

**핵심 문제**:
1. TI 임베딩 파라미터가 생성되지 않음
2. TI optimizer가 생성되지 않음
3. GPT 모델 파라미터가 freeze되지 않음
4. TI optimizer step이 호출되지 않음

**모든 로직을 실제 코드로 구현해야 합니다.**

