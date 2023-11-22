disease_map = {
 '0': '정상',
 # ===== 병해 =====

 # ===== 딸기 =====
 'a1': '잿빛곰팡이병',
 'a2': '흰가루병',
 # ===== 오이 =====
 'a3': '노균병',
 'a4': '흰가루병',
 # ===== 토마토 =====
 'a5': '흰가루병',
 'a6': '잿빛곰팡이병',
 # ===== 고추 =====
 'a7': '탄저병',
 'a8': '흰가루병',

 # ===== 생리장해 =====
 'b1': '냉해피해',
 'b2': '열과',
 'b3': '칼슘결핍',
 'b6': '다량원소결핍(N)',
 'b7': '다량원소결핍(P)',
 'b8': '다량원소결핍(K)'
}

label_for_pepper = {
  0: "0",
  1: "a7",
  2: "a8",
  3: "b1",
  4: "b2",
  5: "b3",
  6: "b6",
  7: "b7",
  8: "b8"
}

def get_disease_map():
    return disease_map

def get_label_map(crop_type):
    # crop_type에 따라 모델 선택
    if crop_type == 'strawberry':
        return
    elif crop_type == 'cucumber':
        return
    elif crop_type == 'tomato':
        return
    elif crop_type == 'pepper':
        return label_for_pepper

def get_model_by_crop(crop_type):
    # crop_type에 따라 모델 선택
    if crop_type == 'strawberry':
        model_type = 'vit_base_patch16_224'
    elif crop_type == 'cucumber':
        model_type = 'vit_base_patch16_224'
    elif crop_type == 'tomato':
        model_type = 'vit_base_patch16_224'
    elif crop_type == 'pepper':
        return 'vit_base_patch16_224'
    else:
        raise ValueError("Invalid crop_type")

def get_weight_by_crop(crop_type):
    # crop_type에 따라 학습시킨 가중치 선택
    if crop_type == 'strawberry':
        return
    elif crop_type == 'cucumber':
        return
    elif crop_type == 'tomato':
        return
    elif crop_type == 'pepper':
        return "vit_classifier_from_timm_mini_pepper.h5"
    else:
        raise ValueError("Invalid crop_type")
