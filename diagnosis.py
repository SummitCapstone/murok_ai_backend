import torch
from torchvision import transforms
from PIL import Image
from utils import *
import timm
import requests
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_classes = 9

logger.debug(f"device: {device}")

def create_model(crop_type):
  logger.debug(f"Creating {crop_type} model...")
  model_type = get_model_by_crop(crop_type)
  logger.debug("Received model_type...")
  model = timm.create_model(model_type, pretrained=False, num_classes=N_classes)
  logger.debug("Received model...")
  WEIGHT_PATH = get_weight_by_crop(crop_type)
  logger.debug("Received weight path...")
  # 모델 가중치 로드
  model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
  logger.debug("Loaded Model...")  
  model.eval()
  logger.debug("Model created and loaded successfully.")
  return model
# Load model in advance
starwberry_model = create_model('strawberry')
cucumber_model = create_model('cucumber')
tomato_model = create_model('tomato')
pepper_model = create_model('pepper')


def load_and_preprocess_image(image_path):
  logger.debug("Loading and preprocessing image...")  
  # 이미지 로드 및 전처리
  image = Image.open(image_path).convert("RGB")
  preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])
  image = preprocess(image)
  image = image.unsqueeze(0)  # 배치 차원 추가
  logger.debug("Image loaded and preprocessed successfully.")
  return image




def diagnose(image_path, crop_type):
  logger.info("Starting diagnosis...")
  start_time = time.time()
  # 이미지 로드 및 전처리
  image = load_and_preprocess_image(image_path)

  # 모델 호출
  if crop_type == 'strawberry':
      model = strawberry_model
  elif crop_type == 'cucumber':
      model = cucumber_model
  elif crop_type == 'tomato':
      model = tomato_model
  elif crop_type == 'pepper':
      model = pepper_model
  else:
      raise ValueError("Invalid crop_type")
      
  # 추론 
  with  torch.no_grad():
    outputs = model(image)
  # 상위 3개의 클래스 및 확률 추출
  _, top_classes = torch.topk(outputs, 3, dim=1)
  top_classes = top_classes.squeeze().tolist()
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
  end_time = time.time()
  elapsed_time = end_time - start_time
  logger.info(f"Diagnosis completed in {elapsed_time} seconds.")
  return top_diseases, top_probabilities
