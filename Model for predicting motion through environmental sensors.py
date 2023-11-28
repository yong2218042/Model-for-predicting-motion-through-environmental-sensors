# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터 불러오기
data_path = r"C:\Users\user\OneDrive\바탕 화면\iot_telemetry_data.csv"
data = pd.read_csv(data_path)

# 데이터 전처리
# (필요에 따라 추가 전처리 수행)
# 예시: 누락된 값 처리, 범주형 변수 처리, 특성 선택 등

# 데이터 열 확인
print(data.columns)

# 범주형 변수 처리
categorical_columns = ['device']  # 다른 범주형 변수가 있는 경우 이곳에 추가

for column in categorical_columns:
    if column in data.columns:
        data = pd.get_dummies(data, columns=[column])

# 데이터 요약 및 시각화
data_summary = data.describe()
print(data_summary)

# 히스토그램 그리기
data.hist(figsize=(10, 10))
plt.show()

# 머신러닝 모델 학습
X = data.drop('motion', axis=1)  # 'motion'을 예측하려면 해당 열을 지정해야 합니다.
y = data['motion']

# 데이터를 학습용과 테스트용으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 성능 평가
y_pred = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 혼동 행렬 출력
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# 정확도를 시각화
plt.bar(['Accuracy'], [accuracy])
plt.ylim(0, 1)
plt.show()

# 추가 결과 확인 여부 입력 받기
while True:
    try:
        user_input = input("더 많은 결과를 확인하시겠습니까? (y/n): ").lower()
        if user_input == 'y':
            # 모델 성능 평가 (다시 예측 및 출력)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy}')

            # 혼동 행렬 출력
            conf_matrix = confusion_matrix(y_test, y_pred)
            print(f'Confusion Matrix:\n{conf_matrix}')

            # 정확도를 시각화
            plt.bar(['Accuracy'], [accuracy])
            plt.ylim(0, 1)
            plt.show()
        elif user_input == 'n':
            print("프로그램을 종료합니다.")
            break
        else:
            print("올바른 입력이 아닙니다. 'y' 또는 'n' 중 하나를 입력하세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
