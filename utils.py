import boto3


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


def generate_labels(prefix, num_labels):
  labels = {0: "0"}
  for i in range(1, 3):
      labels[i] = f"{prefix}{num_labels + i}"
  return labels


# Common labels for all vegetables
common_labels = {3: "b1", 4: "b2", 5: "b3", 6: "b6", 7: "b7", 8: "b8"}


def get_disease_map():
    return disease_map


def get_common_labels():
    return common_labels


def get_label_map(crop_type):
    common_labels = get_common_labels()
    # crop_type에 따라 모델이 학습한 label 선택
    if crop_type == 'strawberry':
        label_for_strawberry = generate_labels("a", 0)  # Assuming 2 additional labels for strawberry
        label_for_strawberry.update(common_labels)
        return label_for_strawberry

    elif crop_type == 'cucumber':
        label_for_cucumber = generate_labels("a", 2)  # Assuming 2 additional labels for cucumber
        label_for_cucumber.update(common_labels)
        return label_for_cucumber

    elif crop_type == 'tomato':
        label_for_tomato = generate_labels("a", 4)  # Assuming 2 additional labels for tomato
        label_for_tomato.update(common_labels)
        return label_for_tomato

    elif crop_type == 'pepper':
        label_for_pepper = generate_labels("a", 6)  # Assuming 2 additional labels for pepper
        label_for_pepper.update(common_labels)
        return label_for_pepper


def download_weights(crop_type):
    # download files from s3
    bucket = 'murok-models'

    if crop_type == 'strawberry':
        file_name = 'weights/strawberry_vit.h5'
        key = 'vit_classifier_from_timm_mini_strawberry.h5'
    elif crop_type == 'cucumber':
        file_name = 'weights/cucumber_vit.h5'
        key = 'vit_classifier_from_timm_mini_cucumber.h5'
    elif crop_type == 'tomato':
        file_name = 'weights/tomato_vit.h5'
        key = 'vit_classifier_from_timm_mini_tomato.h5'
    elif crop_type == 'pepper':
        file_name = 'weights/pepper_vit.h5'
        key = 'vit_classifier_from_timm_mini_pepper.h5'
    else:
        raise ValueError("Invalid crop_type")

    client = boto3.client('s3')
    client.download_file(bucket, key, file_name)


def get_weight_by_crop(crop_type):
    # crop_type에 따라 학습시킨 가중치 선택
    if crop_type == 'strawberry':
        file_name = 'weights/strawberry_vit.h5'
    elif crop_type == 'cucumber':
        file_name = 'weights/cucumber_vit.h5'
    elif crop_type == 'tomato':
        file_name = 'weights/tomato_vit.h5'
    elif crop_type == 'pepper':
        file_name = 'weights/pepper_vit.h5'
    else:
        raise ValueError("Invalid crop_type")
    return file_name
