from config import get_config

import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch


# 한국어 폰트 설정 (Matplotlib에서 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

train_config = get_config()

folder_path = train_config.folder_path

def each_Data(root_dir):
    newlist = []
    data_path = os.path.join(folder_path, root_dir)
    file_list = os.listdir(data_path)

    for file in file_list:
        if 'clean' not in file:
            continue
        elif 'clean' in file:
            file_name = os.path.join(data_path, file)
            data = scipy.io.loadmat(file_name)
            Mydata = (data['eegData_neu'])
            MydataT = Mydata.transpose(2, 0, 1)
        for k in range(MydataT.shape[0]):
            newlist.append(MydataT[k, :, :])

    return newlist

total_list = each_Data('HFsim') + each_Data('LFsim') + each_Data('NH')

# 시간 맞추기
totaltime = []

for r in total_list:
    totaltime.append(r.shape[1])

min_time = min(totaltime)  # 최소시간 = 2007
print(min_time)

# min_time에 맞춰 total 모두 cut
cuttinglist = []

for cut in total_list:
    cutting_data = cut[:, :min_time]
    cuttinglist.append(cutting_data)


for u in range(0, 5):
    print(cuttinglist[u].shape)

# bandpassfilter
# 전극 별로 bandpassfilter를 진행.
def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, signal_data, axis=1)  # axis=0은 ↓, axis=1은 →
    return filtered_data

fs = 250

filtered_data = []

for cut_data in cuttinglist:
    data_array = np.array(cut_data)
    filtered_trial = bandpass_filter(data_array, 13, 30, fs)
    filtered_data.append(filtered_trial)


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


psd_feature = psd_cal(filtered_data, fs, bands)
# ped_feature는 (315,)형태 63채널 * 5 주파수, 4766개가 psd_feature라는 list에 담겨있음.
DATA = np.array(psd_feature)



# 라벨링
num_samples = 4766

labels = np.zeros(num_samples, dtype=int)

labels[1590:3176] = 1
labels[3176:] = 2


from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from statistics import mean

fold_df_clf = SVC(kernel='rbf', random_state=42)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracy = []


for train_idx, test_idx in kfold.split(DATA):
    X_train, X_test = DATA[train_idx], DATA[test_idx]

    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)


    fold_df_clf.fit(X_train_flattened, y_train)

    fold_pred = fold_df_clf.predict(X_test_flattened)

    accuracy = np.round(accuracy_score(y_test, fold_pred), 4)
    cv_accuracy.append(accuracy)  # 10번의 분류 정확도를 리스트에 저장

print(cv_accuracy)
average = mean(cv_accuracy)
acc_std = np.std(cv_accuracy)

print("--- 성능 평가 ---")
print(f"평균 정확도: {average:.4f}")
print(f'표준편차: {acc_std:.4f}')

print(f"\n교차 검증 정확도: {average:.4f} ± {acc_std:.4f}")

print("분류 리포트:")
print(classification_report(y_test, fold_pred, digits=2))

print("혼동 행렬(Confusion Matrix):")
cm = confusion_matrix(y_test, fold_pred)
print(cm)

class_labels = ['HFsim', 'LFsim', 'NH']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_labels, yticklabels=class_labels)

plt.title('Confusion Matrix Heatmap')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()