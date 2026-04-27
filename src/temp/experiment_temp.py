import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ==========================================
# [Hugging Face 공식 문서 반영] timesfm 임포트
# ==========================================
import timesfm

# ==========================================
# 1. 환경 변수 및 설정
# ==========================================
FILE_PATH = './data/ETTm1.csv'
TARGET_COL = 'OT'
C = 96  # Context Length
H = 30  # Horizon Length
S = 1   # Step Size
EPOCHS = 3 # Initial Training 에폭
MAX_TEST_STEPS = None # 전체 실행

# TimesFM 설정
TSFM_MODEL_VER = 'google/timesfm-2.5-200m-pytorch'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        
        # 1. from_pretrained로 모델 로드 (torch_compile 옵션 적용)
        self.tsfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_ver, 
            torch_compile=True if device == "cuda" else False
        )
        
        # 2. ForecastConfig 컴파일 (제공해주신 스니펫 반영)
        self.tsfm.compile(
            timesfm.ForecastConfig(
                max_context=cl,
                max_horizon=hl,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                # ETTm1은 영하 기온 등 음수 값이 존재하므로 False를 추천합니다. (양수만 다루는 도메인이면 True)
                infer_is_positive=False, 
                fix_quantile_crossing=True,
            )
        )
        self.hl = hl

    def predict(self, context_array, horizon):
        """표준 TimesFM 예측 및 Quantile(PI) 추출"""
        
        # 3. 모델 예측 (제공해주신 스니펫 반영)
        point_forecast, quantile_forecast = self.tsfm.forecast(
            horizon=horizon, 
            inputs=[context_array]
        )
        
        # Point Forecast: shape (batch=1, horizon)
        pred_values = point_forecast[0]
        
        # Quantile Forecast: shape (batch=1, horizon, 10) -> [mean, 10th, 20th... 90th]
        quantiles_output = quantile_forecast[0] 
        
        # [중요] 0번 인덱스는 'mean'이므로, 1번 인덱스를 10th(하한)로 사용합니다.
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
        self.arms = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)]
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
    print("1. 데이터 로딩 중...")
    df = pd.read_csv(FILE_PATH)
    target_data = df[TARGET_COL].values
    
    train_split_idx = int(len(target_data) * 0.3)
    train_data = target_data[:train_split_idx]
    test_data = target_data[train_split_idx:]
    
    print("\n2. Transformer 초기 학습 시작...")
    train_dataset = TimeSeriesDataset(train_data, C, H, S)
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
    
    # 모델 래퍼 로드
    model_F = TimesFMWrapper(model_ver=TSFM_MODEL_VER, cl=C, hl=H, device=DEVICE)
    
    extractor = ContextExtractor(window_size=H)
    ensembler = LinUCBWeightEnsembler(context_dim=8, alpha=0.5)
    
    results = []
    prediction_history = {} 
    action_history = {}     
    
    test_steps = len(test_data) - C - H + 1
    if MAX_TEST_STEPS: test_steps = min(test_steps, MAX_TEST_STEPS)

    for t in range(0, test_steps, S):
        current_context = test_data[t : t + C]
        actual_next = test_data[t + C : t + C + H]
        
        # (1) Domain-specific Model (Transformer) 예측
        with torch.no_grad():
            ctx_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_D = model_D(ctx_tensor).squeeze().cpu().numpy()
            
        # (2) Domain-neutral Model (TimesFM) 예측
        pred_F, pi_info = model_F.predict(current_context, H)
        
        prediction_history[t] = {'D': pred_D, 'F': pred_F, 'upper': pi_info['upper'], 'lower': pi_info['lower']}
        
        # (3) Online Learning Logic (지연된 보상 업데이트)
        if t >= H:
            past_actuals = test_data[t + C - H : t + C]
            past_pred_D = prediction_history[t - H]['D']
            past_pred_F = prediction_history[t - H]['F']
            past_upper = prediction_history[t - H]['upper']
            past_lower = prediction_history[t - H]['lower']
            
            x_t = extractor.calculate_xt(past_actuals, past_pred_D, past_pred_F, past_upper, past_lower)
            
            # 이전 행동에 대한 보상 계산 및 업데이트
            if (t - H) in action_history:
                past_arm, past_xt = action_history[t - H]
                past_w_d, past_w_f = ensembler.arms[past_arm]
                past_ensemble = past_w_d * past_pred_D + past_w_f * past_pred_F
                
                mse = np.mean((past_ensemble - past_actuals)**2)
                reward = 1 / (mse + 1e-4)
                ensembler.update(past_arm, past_xt, reward)
            
            # 현재(t) 시점의 가중치 결정
            arm_idx, (w_d, w_f) = ensembler.select_arm(x_t)
            action_history[t] = (arm_idx, x_t)
        else:
            w_d, w_f = 0.5, 0.5
            arm_idx = -1
            
        # (4) 앙상블 결과
        final_pred = w_d * pred_D + w_f * pred_F
        
        results.append({
            't': t,
            'actual': actual_next,
            'pred_Ensemble': final_pred,
            'pred_Transformer': pred_D,
            'pred_TimesFM': pred_F,
            'weight_Transformer': w_d,
            'weight_TimesFM': w_f
        })
        
        if t % 50 == 0:
            print(f"Step {t}/{test_steps} 완료 (가중치 Trans:{w_d:.1f}, TimesFM:{w_f:.1f})")

    # ==========================================
    # 6. 결과 평가 및 시각화
    # ==========================================
    print("\n4. 결과 시각화 중...")
    df_results = pd.DataFrame(results)
    
    df_results['mse_Ensemble'] = df_results.apply(lambda r: np.mean((r['actual'] - r['pred_Ensemble'])**2), axis=1)
    df_results['mse_Transformer'] = df_results.apply(lambda r: np.mean((r['actual'] - r['pred_Transformer'])**2), axis=1)
    df_results['mse_TimesFM'] = df_results.apply(lambda r: np.mean((r['actual'] - r['pred_TimesFM'])**2), axis=1)
    
    print("\n=== 평균 MSE 성능 ===")
    print(f"Transformer Only : {df_results['mse_Transformer'].mean():.4f}")
    print(f"TimesFM Only     : {df_results['mse_TimesFM'].mean():.4f}")
    print(f"LinUCB Ensemble  : {df_results['mse_Ensemble'].mean():.4f}")

    # Plot 1: 가중치 변화 추이
    plt.figure(figsize=(12, 4))
    plt.plot(df_results['t'], df_results['weight_Transformer'], label='Weight (Transformer)', color='blue')
    plt.plot(df_results['t'], df_results['weight_TimesFM'], label='Weight (TimesFM)', color='green')
    plt.title("Dynamic Weight Allocation by LinUCB")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Weight Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Rolling MSE 오차 추이
    window = min(30, len(df_results) // 5)
    plt.figure(figsize=(12, 4))
    plt.plot(df_results['t'], df_results['mse_Ensemble'].rolling(window).mean(), label='Ensemble', color='red', linewidth=2)
    plt.plot(df_results['t'], df_results['mse_Transformer'].rolling(window).mean(), label='Transformer', color='blue', alpha=0.5)
    plt.plot(df_results['t'], df_results['mse_TimesFM'].rolling(window).mean(), label='TimesFM', color='green', alpha=0.5)
    plt.title(f"Rolling MSE Over Time (Window={window})")
    plt.xlabel("Time Step (t)")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()