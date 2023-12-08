import pandas as pd
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

filename = "./data/database.csv"
data = pd.read_csv("./data/database.csv")

# 필요한 열만 추출합니다.
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Timestamp 열을 생성합니다.
timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d + ' ' + t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        timestamp.append('ValueError')

data['Timestamp'] = pd.Series(timestamp).values

# 최종 데이터셋을 생성합니다.
final_data = data[data.Timestamp != 'ValueError'].drop(['Date', 'Time'], axis=1)

# 데이터 분할
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 회귀 모델 생성 및 학습
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# 모델 평가
score = reg.score(X_test, y_test)

# 그리드 서치를 사용한 하이퍼파라미터 튜닝
parameters = {'n_estimators': [10, 20, 50, 100, 200, 500]}
grid_obj = GridSearchCV(reg, parameters)
grid_fit = grid_obj.fit(X_train, y_train)
best_fit = grid_fit.best_estimator_

# 최적화된 모델을 사용한 모델 평가
best_score = best_fit.score(X_test, y_test)

# 결과 출력
print(f"초기 모델 점수: {score}")
print(f"최적 모델 점수: {best_score}")

