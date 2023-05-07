!wget https://bit.ly/fruits_300_data -O fruits_300.npy
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#CSV파일 불러오기
#data = pd.read_csv("")
   
#fish의 종류를 타깃 데이터, 나머지 특성을 입력 데이터
#img_list = data['URL'].to_numpy()
img_list = np.load('fruits_300.npy')
   
#target = data['Class'].to_numpy()
target = np.concatenate(( np.zeros(100), np.ones(100), (np.ones(100)+1) ))
   
#이미지 전처리를 진행하는 함수
def img_preprocessing(img):
    temp = cv2.cvtColor(img, 0)       # 흑백 이미지로 변환
    temp = cv2.resize(temp, (100,100))  # 사이즈 변경
    return temp

#이미지 전처리를 img_list에 적용  
img_list = [img_preprocessing(img) for img in img_list ]
 
#numpy.array로 변
img_list = np.array(img_list)
target = np.array(target)

#1차원배열로 변환
dataset_size = img_list.shape[0]
img_list = img_list.reshape(dataset_size,-1)

#훈련세트와 테스트세트로 나눠주기
train_input, test_input, train_target, test_target = train_test_split(
    img_list, target, random_state=42)

#표준점수로 스케일 조정
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#로지스틱 회귀 모델훈련 및 점수확인
model = LogisticRegression(C=20, max_iter=1000)
model.fit(train_scaled, train_target)
print(model.score(train_scaled, train_target))
print(model.score(test_scaled, test_target))

#결과 확인
i = 0
print(model.predict( [test_scaled[i]] )) # 샘플 5개의 종류 예측
plt.imshow(test_input[i].reshape(100,100,4), cmap='gray_r') 
plt.axis("off")
plt.show()

# 각 Class별 인덱스
index = [ x for x, y in enumerate(model.predict( test_scaled )) if y == 2 ] 

# index에 저장된 번호의 사진을 출력
n = 0;
fig, axs = plt.subplots(5,5,figsize=(5,5))
for i in range(5): 
    for j in range(5): 
        if n >= len(index) : break
        axs[i,j].imshow(test_input[index[i*5+j]].reshape(100,100,4), cmap='gray_r')
        axs[i,j].axis('off') 
        n = n + 1
