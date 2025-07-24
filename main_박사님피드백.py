from config import get_config
from create_dataset  import *

import os
import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 한국어 폰트 설정 (Matplotlib에서 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

train_config = get_config()

folder_path = train_config.folder_path
# folder_path = r'C:\Users\rhs69\Desktop\CONI_hsikr\h_sikr\pythonnew'


dataset=HearingLoss()

train_X, train_y = dataset.get_data()

print(train_X, train_y)
exit()

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
# ------------------------------------
# create_dataset 불러오기
a = timecut(total_list)
filtered_data_result = bandpassing(a, 0.5, 4, fs)
b = final_psd_features = psd_cal(filtered_data_result, fs, bands)
c = labeling()
d, e, f, g = data_split(b, c)


# ------------------------------------

# --- 2. SVM 모델 훈련 ---
# SVM 모델 생성 (기본적인 선형 커널 사용)
model = SVC(kernel='linear', random_state=42)


print('h:',  h.shape)
print('f:', f.shape)
exit()

print("SVM 모델 훈련을 시작합니다...")

# 훈련 데이터로 모델을 학습시킵니다. (fit)
model.fit(h, f)  # 원래는 model.fit(X_train_scaled, Y_train)
print("모델 훈련 완료.\n")


# --- 3. 예측 및 성능 평가 ---
# 학습된 모델로 테스트 데이터에 대한 예측을 수행합니다.
Y_pred = model.predict(i)

# 3-1. 정확도(Accuracy) 평가
accuracy = accuracy_score(g, Y_pred)
print("--- 성능 평가 ---")
print(f"모델의 정확도: {accuracy:.4f}\n")

# 3-2. 분류 리포트(Classification Report) 출력
# 정밀도(Precision), 재현율(Recall), F1-점수(F1-score) 등 더 상세한 지표를 보여줍니다.
print("분류 리포트:")
print(classification_report(g, Y_pred))

# 3-3. 혼동 행렬(Confusion Matrix) 시각화
print("혼동 행렬(Confusion Matrix):")
cm = confusion_matrix(g, Y_pred)
print(cm)

# 혼동 행렬을 히트맵으로 보기 좋게 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(0, 1, 2), yticklabels=np.unique(0, 1, 2))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()