!wget "https://bit.ly/3M9nUaA" -O trash.npy

import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt

#외부파일 불러오기
data = pd.read_csv("https://bit.ly/3LOxQoq")

#data의 URL을 input 데이터 Class를 target 데이터
#input = data['URL'].to_numpy()
input = np.load("trash.npy")
target = data['Class'].to_numpy()

print(pd.unique(data['Class']))


size = 100
#이미지 전처리를 진행하는 함수
def img_preprocessing(img,gray = 1):
    global x
    global y
    temp = cv2.cvtColor(img, 0)       # 흑백 이미지로 
    temp = cv2.resize(temp, (size, size))  # 사이즈 변경
    return temp

#이미지 전처리를 img_list에 적용  
#img_list = [img_preprocessing(img) for img in img_list ]

#img_list = np.array(img_list)
#target = np.array(target)

#필요한 라이브러리 import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#2차원배열로 변환
dataset_size = input.shape[0]
input = input.reshape(dataset_size,-1)

#훈련세트와 테스트세트로 나눠주기
train_input, test_input, train_target, test_target = train_test_split(
    input, target, test_size = 0.1, stratify=target,random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

model = LogisticRegression(C=10, max_iter=1000)
model.fit(train_scaled, train_target)

print(model.score(train_scaled, train_target))
print(model.score(test_scaled, test_target))

index = [ x for x, y in enumerate(model.predict( test_scaled )) if y == '플라스틱' ] # 각 Class별 인덱스

n = 0;
fig, axs = plt.subplots(5,5,figsize=(5,5))
for i in range(5): 
    for j in range(5): 
        if n >= len(index) : break
        axs[i,j].imshow(test_input[index[i*5+j]].reshape(200,200), cmap='gray_r')
        axs[i,j].axis('off') 
        n = n + 1
