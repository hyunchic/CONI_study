import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from warnings import filterwarnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg') # 'Agg' 백엔드 사용
import matplotlib.pyplot as plt

# 불필요한 경고 메시지를 숨깁니다.
filterwarnings('ignore')

# 붓꽃 데이터셋 로드
iris = load_iris()
X = iris.data  # 특징 (Feature) 데이터
y = iris.target # 타겟 (Label) 데이터 (붓꽃의 종)

print(f"특징 데이터(X) 형태: {X.shape}")
print(f"타겟 데이터(y) 형태: {y.shape}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 각 폴드에서의 평가 지표를 저장할 리스트 초기화
accuracies = []
precisions = []
realls = []
f1_scores = []

# StratifiedKFold 객체에서 훈련/테스트 데이터 인덱스를 생성합니다.
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/10 ---")

    # 훈련 세트와 테스트 세트 분할
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # SVM 분류기 모델 생성 및 학습
    # kernel='linear': 선형 SVM을 사용합니다. 간단하고 이해하기 좋습니다.
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    # 테스트 세트에 대한 예측 수행
    y_pred = svm_model.predict(X_test)

    # 평가 지표 계산 및 저장
    # average='macro': 다중 클래스 분류에서 각 클래스에 대한 지표를 계산한 후 평균을 냅니다.
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='macro'))
    realls.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))

    # 현재 폴드의 결과 출력
    print(f"Accuracy: {accuracies[-1]:.4f}")
    print(f"Precision: {precisions[-1]:.4f}")
    print(f"Recall: {realls[-1]:.4f}")
    print(f"F1-Score: {f1_scores[-1]:.4f}")

    # Confusion Matrix 계산
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Confusion Matrix 시각화
    # 붓꽃 데이터셋의 타겟 이름 (클래스 이름)을 설정합니다.
    # iris.target_names는 ['setosa', 'versicolor', 'virginica'] 입니다.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.Blues)  # 색상 맵 설정
    plt.title(f"Confusion Matrix for Fold {fold + 1}")
    plt.show()  # 그래프를 표시합니다.


print("\n--- 최종 결과 요약 ---")

# 각 지표의 평균과 표준 편차 계산 및 출력
# 표준 편차는 소수점 둘째 자리까지, 평균은 넷째 자리까지
print(f"평균 정확도(Accuracy): {np.mean(accuracies):.4f} (표준편차: {np.std(accuracies):.2f})")
print(f"평균 정밀도(Precision): {np.mean(precisions):.4f} (표준편차: {np.std(precisions):.2f})")
print(f"평균 재현율(Recall): {np.mean(realls):.4f} (표준편차: {np.std(realls):.2f})")
print(f"평균 F1-Score: {np.mean(f1_scores):.4f} (표준편차: {np.std(f1_scores):.2f})")