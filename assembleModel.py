import os
import cv2
import numpy as np
import torch
import pickle
from transformers import AutoModelForImageClassification, EfficientNetImageProcessor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Fourier_model = AutoModelForImageClassification.from_pretrained("./Models_bk/Fourier_Model")
image_processor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
def predict_fourier(image):
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #if img is None:
    #    print(f"Error loading image: {image_path}")
    #    return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    #magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(magnitude_spectrum)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #model = AutoModelForImageClassification.from_pretrained("./Models_bk/Fourier_Model")
    model = Fourier_model
    #image_processor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
    inputs = image_processor(images=image, return_tensors="pt")
    labels = torch.tensor([1])
    model.eval()
    outputs = model(**inputs, labels=labels)
    return outputs.loss.item()

Amp_model = AutoModelForImageClassification.from_pretrained("./Models_bk/Amplify_Model")
def predict_amplify(image):
    #img = cv2.imread(image_path)
    #if img is None:
    #    print(f"Error loading image: {image_path}")
    #    return None
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #model = AutoModelForImageClassification.from_pretrained("./Models_bk/Amplify_Model")
    model = Amp_model
    #image_processor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
    inputs = image_processor(images=image, return_tensors="pt")
    labels = torch.tensor([1])
    model.eval()
    outputs = model(**inputs, labels=labels)
    return outputs.loss.item()

Original_model = AutoModelForImageClassification.from_pretrained("./Models_bk/Original_Model")
def predict_original(image):
    #model = AutoModelForImageClassification.from_pretrained("./Models_bk/Original_Model")
    model = Original_model
    #image_processor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
    #image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = image_processor(images=image, return_tensors="pt")
    labels = torch.tensor([1])
    model.eval()
    outputs = model(**inputs, labels=labels)
    return outputs.loss.item()

def get_data(image):
    original = predict_original(image.copy())
    amplify = predict_amplify(image.copy())
    fourier = predict_fourier(image.copy())
    return [original, amplify, fourier]

def predict(image_path):
    loss_values = get_data(image_path)
    with open('classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    f.close()
    result = classifier.predict([loss_values])
    if result == 0:
        return 'Deepfake'
    else:
        return 'Real'

def main():
    dataset = []
    labels = []

    print('Start Load Dataset')

    # 데이터 로드 및 특징 추출
    fake_data_path = "./val/vali_Deepfake"
    for file_name in os.listdir(fake_data_path):
        file_path = os.path.join(fake_data_path, file_name)
        loss_values = [
            predict_original(file_path),
            predict_amplify(file_path),
            predict_fourier(file_path)
        ]
        if None not in loss_values:
            dataset.append(loss_values)
            labels.append(0)  # Deepfake label

    '''
    spoofing_data_path = "./rootData_original/FaceSpoofing"
    for file_name in os.listdir(spoofing_data_path):
        file_path = os.path.join(spoofing_data_path, file_name)
        loss_values = [
            predict_original(file_path),
            predict_amplify(file_path),
            predict_fourier(file_path)
        ]
        if None not in loss_values:
            dataset.append(loss_values)
            labels.append(1)  # FaceSpoofing label
    '''

    #real_data_path = "./rootData_original/Real"
    real_data_path = "./val/vali_Real"
    for file_name in os.listdir(real_data_path):
        file_path = os.path.join(real_data_path, file_name)
        loss_values = [
            predict_original(file_path),
            predict_amplify(file_path),
            predict_fourier(file_path)
        ]
        if None not in loss_values:
            dataset.append(loss_values)
            labels.append(1)  # Real label

    print('Complete Load')

    print('Start Model fit')

    # 학습 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=156)

    # 개별 모델 정의
    lr_clf = LogisticRegression()
    knn_clf = KNeighborsClassifier(n_neighbors=8)

    # Voting Classifier 정의 및 학습
    vo_clf = VotingClassifier(estimators=[('LR', lr_clf), ('KNN', knn_clf)], voting='soft')
    vo_clf.fit(X_train, y_train)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(vo_clf, f)
    f.close()
    pred = vo_clf.predict(X_test)
    print(f'Voting 분류기 정확도: {accuracy_score(y_test, pred):.4f}')

    '''
    # 개별 모델 평가
    classifiers = [lr_clf, knn_clf]
    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
        class_name = classifier.__class__.__name__
        print(f'{class_name} 정확도: {accuracy_score(y_test, pred):.4f}')
    '''

if __name__ == "__main__":
    main()
