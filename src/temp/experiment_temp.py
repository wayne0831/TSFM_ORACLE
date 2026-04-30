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
# 0. 재현성을 위한 시드 고정 및 온라인 스케일러
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

class OnlineScaler:
    """실시간으로 평균과 분산을 업데이트하여 데이터를 표준화(Z-score)함"""
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def transform(self, x):
        if self.n < 2:
            return x
        std = np.sqrt(self.m2 / (self.n - 1))
        # 0으로 나누기 방지
        return (x - self.mean) / (std + 1e-8)

# ==========================================
# 1. 환경 변수 및 설정
# ==========================================
FILE_PATH = './data/ETTh1.csv'
TARGET_COL = 'OT'
C = 96  # Context Length
H = 30  # Horizon Length
S = 15  # Step Size 
EPOCHS = 20 # Initial Training 에폭 (성능 개선을 위해 상향)
MAX_TEST_STEPS = None 
SEED = 42 
ALPHA = 0.2 # 탐색 억제

# TimesFM 설정
TSFM_MODEL_VER = 'google/timesfm-2.5-200m-pytorch'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 결과 저장 경로
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
    def __init__(self, model_ver, cl=96, hl=30, device="cuda"):
        print(f"Loading TimesFM: {model_ver} on {device}...")
        self.tsfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_ver, torch_compile=True if device == "cuda" else False
        )
        self.tsfm.compile(
            timesfm.ForecastConfig(
                max_context=cl, max_horizon=hl, normalize_inputs=True,
                use_continuous_quantile_head=True, force_flip_invariance=True,
                infer_is_positive=False, fix_quantile_crossing=True,
            )
        )
        self.hl = hl

    def predict(self, context_array, horizon):
        point_forecast, quantile_forecast = self.tsfm.forecast(horizon=horizon, inputs=[context_array])
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
        #return np.array([mse_d, mse_f, rel_err])


class LinUCBWeightEnsembler:
    def __init__(self, context_dim=8, alpha=0.1):
        self.alpha = alpha
        self.d = context_dim
        self.arms = [(round(1.0 - i * 0.1, 1), round(i * 0.1, 1)) for i in range(11)]        
        self.A = [np.eye(self.d) for _ in range(len(self.arms))]
        self.b = [np.zeros((self.d, 1)) for _ in range(len(self.arms))]
        # 온라인 스케일러 추가
        self.scaler = OnlineScaler(dim=self.d)

    def select_arm(self, x_t):
        self.scaler.update(x_t)
        scaled_xt = self.scaler.transform(x_t)
        
        x = scaled_xt.reshape(-1, 1)
        p_t = []
        for i in range(len(self.arms)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            ucb_score = (theta.T @ x) + self.alpha * np.sqrt(x.T @ A_inv @ x)
            p_t.append(ucb_score.item())
        best_arm_idx = np.argmax(p_t)
        return best_arm_idx, self.arms[best_arm_idx]

    def update(self, arm_idx, x_t, reward):
        scaled_xt = self.scaler.transform(x_t)
        x = scaled_xt.reshape(-1, 1)
        self.A[arm_idx] += x @ x.T
        self.b[arm_idx] += reward * x

# ==========================================
# 5. 메인 파이프라인 실행
# ==========================================
def main():
    os.makedirs(PRED_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True); os.makedirs(PERF_DIR, exist_ok=True)
    set_seed(SEED)

    print("1. 데이터 로딩 중...")
    df = pd.read_csv(FILE_PATH)
    target_data = df[TARGET_COL].values
    train_split_idx = int(len(target_data) * 0.1)
    train_data = target_data[:train_split_idx]
    test_data = target_data[train_split_idx:]
    
    print(f"\n2. Transformer 초기 학습 시작... (Epochs: {EPOCHS})")
    train_dataset = TimeSeriesDataset(train_data, C, H, 1) 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model_D = SimpleTransformer(C, H).to(DEVICE)
    optimizer = torch.optim.Adam(model_D.parameters(), lr=0.001); criterion = nn.MSELoss()
    
    model_D.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(); preds = model_D(batch_x); loss = criterion(preds, batch_y); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
    
    print("\n3. 온라인 예측 및 LinUCB 평가 시작 (Scaling Enabled)...")
    model_D.eval(); model_F = TimesFMWrapper(TSFM_MODEL_VER, C, H, DEVICE)
    extractor = ContextExtractor(H)
    ensembler = LinUCBWeightEnsembler(8, ALPHA)
    
    weight_history, prediction_history, action_history = [], {}, {}
    test_steps = len(test_data) - C - H + 1
    if MAX_TEST_STEPS: test_steps = min(test_steps, MAX_TEST_STEPS)

    test_len = len(test_data)
    timeline_ensemble = np.full(test_len, np.nan); timeline_transformer = np.full(test_len, np.nan); timeline_timesfm = np.full(test_len, np.nan)

    for t in range(0, test_steps, S):
        current_context = test_data[t : t + C]; actual_next = test_data[t + C : t + C + H]
        with torch.no_grad():
            ctx_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_D = model_D(ctx_tensor).squeeze().cpu().numpy()
        pred_F, pi_info = model_F.predict(current_context, H)
        prediction_history[t] = {'D': pred_D, 'F': pred_F, 'upper': pi_info['upper'], 'lower': pi_info['lower']}
        
        if t >= H:
            past_actuals = test_data[t + C - H : t + C]
            past_pred_D, past_pred_F = prediction_history[t - H]['D'], prediction_history[t - H]['F']
            past_upper, past_lower = prediction_history[t - H]['upper'], prediction_history[t - H]['lower']
            x_t = extractor.calculate_xt(past_actuals, past_pred_D, past_pred_F, past_upper, past_lower)
            
            if (t - H) in action_history:
                past_arm, past_xt = action_history[t - H]
                past_w_d, past_w_f = ensembler.arms[past_arm]
                mse = np.mean(((past_w_d * past_pred_D + past_w_f * past_pred_F) - past_actuals)**2)
                ensembler.update(past_arm, past_xt, 1 / (mse + 1e-4))
            
            arm_idx, (w_d, w_f) = ensembler.select_arm(x_t)
            action_history[t] = (arm_idx, x_t)
        else:
            w_d, w_f = 0.5, 0.5; arm_idx = -1
            
        timeline_ensemble[t+C:t+C+H] = w_d * pred_D + w_f * pred_F
        timeline_transformer[t+C:t+C+H] = pred_D; timeline_timesfm[t+C:t+C+H] = pred_F
        weight_history.append({'t': t, 'w_f': w_f})
        if t % (S * 20) == 0: print(f"Step {t}/{test_steps} (TimesFM Weight:{w_f:.2f})")

    # ==========================================
    # 6. 결과 저장 및 시각화
    # ==========================================
    valid_idx = ~np.isnan(timeline_ensemble)
    actual_valid, ens_valid, trans_valid, tsfm_valid = test_data[valid_idx], timeline_ensemble[valid_idx], timeline_transformer[valid_idx], timeline_timesfm[valid_idx]
    
    mse_t, mse_f, mse_e = np.mean((actual_valid - trans_valid)**2), np.mean((actual_valid - tsfm_valid)**2), np.mean((actual_valid - ens_valid)**2)
    print(f"\n[Final MSE] Trans: {mse_t:.4f}, TimesFM: {mse_f:.4f}, Ensemble: {mse_e:.4f}")

    # Plot 1: Weight History
    df_w = pd.DataFrame(weight_history)
    plt.figure(figsize=(12, 4)); plt.plot(df_w['t'], df_w['w_f'], color='green', marker='o', markersize=2)
    plt.title("Dynamic Weight Allocation (TimesFM Weight)"); plt.ylim(-0.05, 1.05); plt.grid(True); plt.savefig(f"{PLOT_DIR}/weight_ETTh1.png"); plt.show()

    # Plot 2: Rolling MSE
    se_df = pd.DataFrame({'Ensemble': (actual_valid - ens_valid)**2, 'Transformer': (actual_valid - trans_valid)**2, 'TimesFM': (actual_valid - tsfm_valid)**2})
    window = min(100, len(se_df) // 10)
    plt.figure(figsize=(12, 4)); plt.plot(se_df['Ensemble'].rolling(window).mean(), label='Ensemble', color='red')
    plt.plot(se_df['Transformer'].rolling(window).mean(), label='Transformer', color='blue', alpha=0.3); plt.plot(se_df['TimesFM'].rolling(window).mean(), label='TimesFM', color='green', alpha=0.3)
    plt.title(f"Rolling Squared Error (Window={window})"); plt.legend(); plt.grid(True); plt.savefig(f"{PLOT_DIR}/rolling_mse_ETTh1.png"); plt.show()

    # Plot 3: Prediction Timeline
    plt.figure(figsize=(14, 5)); plt.plot(test_data, label='Actual', color='black', alpha=0.8)
    plt.plot(timeline_transformer, label='Transformer', color='blue', alpha=0.2); plt.plot(timeline_timesfm, label='TimesFM', color='green', alpha=0.4)
    plt.plot(timeline_ensemble, label='Ensemble', color='red', linewidth=1.5)
    plt.title("Prediction Timeline Comparison"); plt.legend(); plt.grid(True); plt.savefig(f"{PLOT_DIR}/prediction_timeline_ETTh1.png"); plt.show()

if __name__ == "__main__":
    main()