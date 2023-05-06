import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
   
#CSV파일 불러오기
#data = pd.read_csv("http://bit.ly/fish_csv_data")
   
#사진의 종류를 타깃 데이터, 나머지 특성을 입력 데이터
#img_list = data['URL'].to_numpy()
img_list = np.load('fruits_300.npy')
   
#target = data['Class'].to_numpy()
target = np.concatenate(( np.zeros(100), np.ones(100), (np.ones(100)+1) ))
   
#이미지 전처리를 진행하는 함수
def img_preprocessing(img):
  temp = cv2.cvtColor(img, 0)       # 흑백 이미지로 로드
  temp = cv2.resize(temp, (100,100))  # 사이즈 변경
  return temp
   
#이미지 출력
for img in img_list[:2]
   img = img_preprocessing(img)
   plt.imshow(img, cmap='gray_r') 
   plt.axis("off")
   plt.show()
