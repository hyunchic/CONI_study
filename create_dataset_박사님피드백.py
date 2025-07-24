from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HearingLoss:
    def __init__(self):
        pass
    def get_data(self):
        #change to return X_train_scaled, X_test_scaled
        return 1, 1

def timecut(full_list):
    totaltime = []

    for r in full_list:
        totaltime.append(r.shape[1])

    min_time = min(totaltime)  # 최소시간 = 2007
    print(min_time)

    # min_time에 맞춰 total 모두 cut
    cuttinglist = []

    for cut in full_list:
        cutting_data = cut[:, :min_time]
        cuttinglist.append(cutting_data)

    for u in range(0, 5):
        print(cuttinglist[u].shape)

    return cuttinglist


fs = 250

def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, signal_data, axis=1)  # axis=0은 ↓, axis=1은 →
    return filtered_data


def bandpassing(data_list, lowcut, highcut, fs):  # 실제 실행에서는 filtered_data_result = bandpassing(cuttinglist, 30, 45, fs)
    filtering_data = []
    for cut_data in data_list:
        data_array = np.array(cut_data)
        filtered_trial = bandpass_filter(data_array, lowcut, highcut, fs)
        filtering_data.append(filtered_trial)

    return filtering_data

bands = {
    'Delta': [0.5, 4],
    'Theta': [4, 7],
    'Alpha': [8, 12],
    'Beta': [13, 30],
    'Gamma': [30, 45]
}

def extract_psd_features(trial_data, fs, bands):
    n_channels, _ = trial_data.shape
    features = np.zeros((n_channels, len(bands)))
    for ch_idx in range(n_channels):
        freqs, psd = welch(trial_data[ch_idx, :], fs=fs, nperseg=fs)
        for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
            idx_band = np.where((freqs >= low) & (freqs <= high))[0]
            features[ch_idx, band_idx] = np.mean(psd[idx_band])
    return features


def psd_cal(filtered_data_list, fs, bands):  # 실제 실행에서는 final_psd_features = psd_cal(filtered_data_result, fs, bands)
    X_features_list = []
    for trial_data in filtered_data_list:
        # 1. trial 데이터에서 (63, 5) 모양의 PSD 특징 추출
        psd_feature_matrix = extract_psd_features(trial_data, fs, bands)

        # 2. SVM 입력을 위해 2D 특징을 1D 벡터로 펼치기 (Flattening)
        # (63, 5) -> (63 * 5) = (315,) 모양의 1차원 벡터로 변환
        psd_feature_vector = psd_feature_matrix.flatten()

        X_features_list.append(psd_feature_vector)

    return X_features_list


def labeling():
    num_samples = 4766

    labels = np.zeros(num_samples, dtype=int)

    labels[1590:3176] = 1
    labels[3176:] = 2

    return labels


def data_split(X_featurelist, LABELS):  # 데이터 분할 train, test
    X = np.array(X_featurelist)

    Y = np.array(LABELS)

    print(f'원본데이터(X) 형태: {X.shape}')
    print(f'원본 라벨(Y) 형태: {Y.shape}\n')

    # 8:2 데이터 분할
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )

    # --- 1. 데이터 전처리 ---
    # 1-1. 3D 데이터를 2D로 펼치기 (Flattening)
    # (샘플 수, 63, 2007) -> (샘플 수, 63 * 2007)
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train_samples, -1)
    X_test_flat = X_test.reshape(n_test_samples, -1)

    print(f"원본 훈련 데이터 형태: {X_train.shape}")
    print(f"2D로 변환된 훈련 데이터 형태: {X_train_flat.shape}\n")

    return X_train_flat, X_test_flat, Y_train, Y_test


def data_scaling(X_trainF, XtestF):
    # 1-2. 데이터 스케일링 (Standardization)
    scaler = StandardScaler()

    # 중요: scaler는 반드시 '훈련 데이터'에만 fit 해야 합니다.
    # (테스트 데이터의 정보가 훈련 과정에 유출되는 것을 막기 위함)
    scaler.fit(X_trainF)

    # 훈련 데이터와 테스트 데이터 모두에 동일한 스케일러를 적용합니다.
    X_train_scaled = scaler.transform(X_trainF)
    X_test_scaled = scaler.transform(XtestF)

    print("데이터 스케일링 완료.\n")

    return X_train_scaled, X_test_scaled