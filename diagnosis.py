import torch
from torchvision import transforms
from PIL import Image
from utils import *
import timm
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_classes = 9

def load_and_preprocess_image(image_path):
  # 이미지 로드 및 전처리
  image = Image.open(image_path).convert("RGB")
  preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])
  image = preprocess(image)
  image = image.unsqueeze(0)  # 배치 차원 추가
  return image


def create_model(crop_type):
  model_type = get_model_by_crop(crop_type)
  model = timm.create_model(model_type, pretrained=False, num_classes=N_classes)
  WEIGHT_PATH = get_weight_by_crop(crop_type)

  # 모델 가중치 저장
  torch.save(model.state_dict(), WEIGHT_PATH)

  # 모델 가중치 로드
  model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
  model.eval()

  return model


def diagnose(image_path, crop_type):
  # 이미지 로드 및 전처리
  image = load_and_preprocess_image(image_path)

  # 모델 호출
  model = create_model(crop_type)

  # 추론
  with torch.no_grad():
    outputs = model(image)
  print(outputs)
  # 상위 3개의 클래스 및 확률 추출
  _, top_classes = torch.topk(outputs, 3, dim=1)
  print(top_classes)
  top_classes = top_classes.squeeze().tolist()
  print(top_classes)
  # 클래스를 병명으로 변환
  disease_map = get_disease_map()
  label_map = get_label_map(crop_type)
  # 병명 코드 구하기
  code_list = [label_map[i] for i in top_classes]
  # 병명 코드로부터 병명 구하기
  top_diseases = [disease_map[code] for code in code_list]

  # 각 클래스의 확률값 추출
  softmax = torch.nn.Softmax(dim=1)
  probabilities = softmax(outputs)[0].tolist()
  top_probabilities = [probabilities[class_label] for class_label in top_classes]

  return top_diseases, top_probabilities
