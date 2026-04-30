import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random

# ==========================================
# [Hugging Face 공식 문서 반영] timesfm 임포트
# ==========================================
import timesfm

# ==========================================
# 0. 재현성을 위한 시드 고정 함수
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 환경 변수 및 설정
# ==========================================
FILE_PATH = './data/ETTh1.csv'
TARGET_COL = 'OT'
C = 96  # Context Length
H = 30  # Horizon Length
S = 15  # Step Size 
EPOCHS = 20 # Initial Training 에폭
MAX_TEST_STEPS = None # 전체 실행
SEED = 42 # 시드 값 설정

ALPHA = 0.1

# TimesFM 설정
TSFM_MODEL_VER = 'google/timesfm-2.5-200m-pytorch'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 결과 저장 경로 설정 (프로젝트 구조 반영)
PRED_DIR = './results/predictions/Ensemble'
PLOT_DIR = './results/plot/Ensemble'
PERF_DIR = './results/performance/Ensemble'

# ==========================================
# 2. 데이터셋 클래스 및 로딩
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_len, horizon, step):
        self.data = data
        self.c    = context_len
        self.h    = horizon
        self.s    = step
        self.indices = range(0, len(data) - context_len - horizon + 1, step)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        start = self.indices[i]
        x = self.data[start : start + self.c]
        y = self.data[start + self.c : start + self.c + self.h]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ==========================================
# 3. 모델 정의
# ==========================================
class SimpleTransformer(nn.Module):
    """Domain-specific Model (Vanilla Transformer)"""
    def __init__(self, context_len=96, horizon=30):
        super().__init__()
        self.linear_in = nn.Linear(1, 16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear_out = nn.Linear(16 * context_len, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear_in(x)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.linear_out(x)


class TimesFMWrapper:
    """Domain-neutral Model (TimesFM 2.5 공식 스니펫 기반 래퍼)"""
    def __init__(self, model_ver, cl=96, hl=30, device="cuda"):
        print(f"Loading TimesFM: {model_ver} on {device}...")
        
        self.tsfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_ver, 
            torch_compile=True if device == "cuda" else False
        )
        
        self.tsfm.compile(
            timesfm.ForecastConfig(
                max_context=cl,
                max_horizon=hl,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False, 
                fix_quantile_crossing=True,
            )
        )
        self.hl = hl

    def predict(self, context_array, horizon):
        """표준 TimesFM 예측 및 Quantile(PI) 추출"""
        point_forecast, quantile_forecast = self.tsfm.forecast(
            horizon=horizon, 
            inputs=[context_array]
        )
        
        pred_values = point_forecast[0]
        quantiles_output = quantile_forecast[0] 
        
        pi_lower = quantiles_output[:, 1]
        pi_upper = quantiles_output[:, -1]
        
        return pred_values, {'upper': pi_upper, 'lower': pi_lower}


# ==========================================
# 4. 프레임워크 핵심 클래스 (Context & LinUCB)
# ==========================================
class ContextExtractor:
    def __init__(self, window_size=30, eps=1e-6):
        self.H = window_size
        self.eps = eps

    def calculate_xt(self, y_true, pred_D, pred_F, pi_upper_F, pi_lower_F):
        widths = pi_upper_F - pi_lower_F
        mu_width = np.mean(widths)
        sigma_width = np.std(widths)
        cv_pi = sigma_width / (mu_width + self.eps)

        severity_list = []
        for i in range(len(y_true)):
            if y_true[i] > pi_upper_F[i]:
                sev = abs(y_true[i] - pi_upper_F[i]) / (widths[i] + self.eps)
            elif y_true[i] < pi_lower_F[i]:
                sev = abs(y_true[i] - pi_lower_F[i]) / (widths[i] + self.eps)
            else:
                sev = 0
            severity_list.append(sev)
        avg_severity = np.mean(severity_list)

        mu_t = np.mean(y_true)
        sigma_t = np.std(y_true)
        volatility = sigma_t / (mu_t + self.eps)

        mse_d = np.mean((y_true - pred_D)**2)
        mse_f = np.mean((y_true - pred_F)**2)
        rel_err = mse_f / (mse_d + self.eps)

        return np.array([cv_pi, avg_severity, mu_t, sigma_t, volatility, mse_d, mse_f, rel_err])

class LinUCBWeightEnsembler:
    def __init__(self, context_dim=8, alpha=0.5):
        self.alpha = alpha
        self.d = context_dim
        # 0.1 간격 11개 Arms 구성
        self.arms = [
            (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), 
            (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), 
            (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)
        ]        
        self.A = [np.eye(self.d) for _ in range(len(self.arms))]
        self.b = [np.zeros((self.d, 1)) for _ in range(len(self.arms))]

    def select_arm(self, x_t):
        x = x_t.reshape(-1, 1)
        p_t = []
        for i in range(len(self.arms)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            ucb_score = (theta.T @ x) + self.alpha * np.sqrt(x.T @ A_inv @ x)
            p_t.append(ucb_score.item())
        best_arm_idx = np.argmax(p_t)
        return best_arm_idx, self.arms[best_arm_idx]

    def update(self, arm_idx, x_t, reward):
        x = x_t.reshape(-1, 1)
        self.A[arm_idx] += x @ x.T
        self.b[arm_idx] += reward * x

# ==========================================
# 5. 메인 파이프라인 실행
# ==========================================
def main():
    # 저장 디렉토리 생성
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(PERF_DIR, exist_ok=True)

    # 시드 고정
    set_seed(SEED)

    print("1. 데이터 로딩 중...")
    df = pd.read_csv(FILE_PATH)
    target_data = df[TARGET_COL].values
    
    train_split_idx = int(len(target_data) * 0.1)
    train_data = target_data[:train_split_idx]
    test_data = target_data[train_split_idx:]
    
    print("\n2. Transformer 초기 학습 시작... (Seed: 42)")
    train_dataset = TimeSeriesDataset(train_data, C, H, 1) 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model_D = SimpleTransformer(C, H).to(DEVICE)
    optimizer = torch.optim.Adam(model_D.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model_D.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model_D(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
    
    print("\n3. 온라인 예측 및 LinUCB 평가 시작...")
    model_D.eval()
    model_F = TimesFMWrapper(model_ver=TSFM_MODEL_VER, cl=C, hl=H, device=DEVICE)
    extractor = ContextExtractor(window_size=H)
    ensembler = LinUCBWeightEnsembler(context_dim=8, alpha=ALPHA)
    
    # 가중치 기록용
    weight_history = []
    prediction_history = {} 
    action_history = {}     
    
    test_steps = len(test_data) - C - H + 1
    if MAX_TEST_STEPS: test_steps = min(test_steps, MAX_TEST_STEPS)

    test_len = len(test_data)
    timeline_ensemble = np.full(test_len, np.nan)
    timeline_transformer = np.full(test_len, np.nan)
    timeline_timesfm = np.full(test_len, np.nan)
    
    # 가중치 타임라인 배열 (CSV 저장용)
    timeline_weight = np.full(test_len, np.nan)

    for t in range(0, test_steps, S):
        current_context = test_data[t : t + C]
        actual_next = test_data[t + C : t + C + H]
        
        # (1) 예측 수행
        with torch.no_grad():
            ctx_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_D = model_D(ctx_tensor).squeeze().cpu().numpy()
            
        pred_F, pi_info = model_F.predict(current_context, H)
        prediction_history[t] = {'D': pred_D, 'F': pred_F, 'upper': pi_info['upper'], 'lower': pi_info['lower']}
        
        # (2) LinUCB 보상 업데이트 및 가중치 결정
        if t >= H:
            past_actuals = test_data[t + C - H : t + C]
            past_pred_D = prediction_history[t - H]['D']
            past_pred_F = prediction_history[t - H]['F']
            past_upper = prediction_history[t - H]['upper']
            past_lower = prediction_history[t - H]['lower']
            
            x_t = extractor.calculate_xt(past_actuals, past_pred_D, past_pred_F, past_upper, past_lower)
            
            if (t - H) in action_history:
                past_arm, past_xt = action_history[t - H]
                past_w_d, past_w_f = ensembler.arms[past_arm]
                past_ensemble = past_w_d * past_pred_D + past_w_f * past_pred_F
                
                mse = np.mean((past_ensemble - past_actuals)**2)
                reward = 1 / (mse + 1e-4)
                ensembler.update(past_arm, past_xt, reward)
            
            arm_idx, (w_d, w_f) = ensembler.select_arm(x_t)
            action_history[t] = (arm_idx, x_t)
        else:
            w_d, w_f = 0.5, 0.5
            arm_idx = -1
            
        final_pred = w_d * pred_D + w_f * pred_F
        
        # (3) 배열 업데이트 (최신 예측값 덮어쓰기)
        idx_start = t + C
        idx_end = t + C + H
        
        timeline_ensemble[idx_start:idx_end] = final_pred
        timeline_transformer[idx_start:idx_end] = pred_D
        timeline_timesfm[idx_start:idx_end] = pred_F
        
        # 해당 구간에 적용된 TimesFM 가중치 덮어쓰기
        timeline_weight[idx_start:idx_end] = w_f
        
        # 가중치 시각화를 위한 저장 (Plot 1 용도)
        weight_history.append({'t': t, 'w_f': w_f})
        
        if t % (S * 10) == 0: 
            print(f"Step {t}/{test_steps} 완료 (가중치 TimesFM:{w_f:.1f})")

    # ==========================================
    # 6. 결과 저장 및 평가 시각화
    # ==========================================
    print("\n4. 결과 저장 및 시각화 중...")
    
    # [저장 1] 예측값 CSV 저장
    df_preds = pd.DataFrame({
        'Actual': test_data,
        'Transformer': timeline_transformer,
        'TimesFM': timeline_timesfm,
        'Ensemble': timeline_ensemble,
        'TimesFM_Weight': timeline_weight # CSV에도 가중치 타임라인 포함
    })
    pred_file_path = f"{PRED_DIR}/predictions_ETTh1.csv"
    df_preds.to_csv(pred_file_path, index=False)
    print(f" - 예측 데이터 저장 완료: {pred_file_path}")

    # 유효한(NaN이 아닌) 평가 구간 필터링
    valid_idx = ~np.isnan(timeline_ensemble)
    actual_valid = test_data[valid_idx]
    ens_valid = timeline_ensemble[valid_idx]
    trans_valid = timeline_transformer[valid_idx]
    tsfm_valid = timeline_timesfm[valid_idx]
    
    mse_trans = np.mean((actual_valid - trans_valid)**2)
    mse_tsfm = np.mean((actual_valid - tsfm_valid)**2)
    mse_ens = np.mean((actual_valid - ens_valid)**2)

    # [저장 2] 성능 지표(MSE) 저장
    perf_file_path = f"{PERF_DIR}/metrics_ETTh1.txt"
    with open(perf_file_path, "w") as f:
        f.write("=== 최신 덮어쓰기 기반 (단일 타임라인) 평균 MSE ===\n")
        f.write(f"Transformer Only : {mse_trans:.4f}\n")
        f.write(f"TimesFM Only     : {mse_tsfm:.4f}\n")
        f.write(f"LinUCB Ensemble  : {mse_ens:.4f}\n")
    print(f" - 성능 평가 저장 완료: {perf_file_path}")

    print(f"\nTransformer Only : {mse_trans:.4f}")
    print(f"TimesFM Only     : {mse_tsfm:.4f}")
    print(f"LinUCB Ensemble  : {mse_ens:.4f}")

    # [저장 3] Plot 저장 및 시각화
    df_weights = pd.DataFrame(weight_history)
    
    # Plot 1: TimesFM 가중치 변화 추이
    plt.figure(figsize=(12, 4))
    plt.plot(df_weights['t'], df_weights['w_f'], label='Weight (TimesFM)', color='green', marker='o', markersize=2)
    plt.title("Dynamic Weight Allocation by LinUCB (TimesFM Weight)")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Weight Ratio (0.0 to 1.0)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/weight_allocation_ETTh1.png")
    plt.show()

    # Plot 2: 타임라인별 Squared Error 흐름 (Rolling)
    se_df = pd.DataFrame({
        'Ensemble': (actual_valid - ens_valid)**2,
        'Transformer': (actual_valid - trans_valid)**2,
        'TimesFM': (actual_valid - tsfm_valid)**2
    })
    
    window = min(100, len(se_df) // 10)
    plt.figure(figsize=(12, 4))
    plt.plot(se_df['Ensemble'].rolling(window).mean(), label='Ensemble', color='red', linewidth=2)
    plt.plot(se_df['Transformer'].rolling(window).mean(), label='Transformer', color='blue', alpha=0.5)
    plt.plot(se_df['TimesFM'].rolling(window).mean(), label='TimesFM', color='green', alpha=0.5)
    plt.title(f"Rolling Squared Error Over Time (Window={window}, S={S})")
    plt.xlabel("Data Index")
    plt.ylabel("Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/rolling_mse_ETTh1.png")
    plt.show()
    
    # ---------------------------------------------------------
    # [수정] Plot 3: 예측값 타임라인 시각화 (단일 y축 복구)
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 5))
    
    # 실제값 (가장 뒤에 깔리도록 zorder=1)
    plt.plot(test_data, label='Actual (y)', color='black', linewidth=1.5, zorder=1)
    
    # 모델별 예측값
    plt.plot(timeline_transformer, label='Transformer', color='blue', alpha=0.4, zorder=2)
    plt.plot(timeline_timesfm, label='TimesFM', color='green', alpha=0.6, zorder=3)
    
    # Ensemble 값은 가장 중요하므로 두껍게(linewidth) 위로(zorder) 올림
    plt.plot(timeline_ensemble, label='Ensemble', color='red', linewidth=2.0, zorder=4)
    
    plt.title("Prediction Timeline: Actual vs Models (Latest Update)")
    plt.xlabel("Time Step")
    plt.ylabel("Value (OT)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # Plot 3 이미지 저장
    plt.savefig(f"{PLOT_DIR}/prediction_timeline_ETTh1.png")
    plt.show()
    
    print("\n모든 실험 및 저장 절차가 완료되었습니다!")

if __name__ == "__main__":
    main()