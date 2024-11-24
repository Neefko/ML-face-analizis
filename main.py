from deepface import DeepFace
import json
import numpy as np


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def face_verification(img1, img2):
    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2)
        with open('result.json', 'w') as f:
            json.dump(result, f, default=convert_to_serializable)
        return result
    except Exception as e:
        print(f"Error in face_verification: {e}")


def face_analysis(img):
    try:
        result = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'race', 'emotion'])
        # Коррекция возраста
        if result[0]['age'] < 20:
            result[0]['age'] -= 10

        with open('analysis.json', 'w') as f:
            json.dump(result, f, default=convert_to_serializable)
        return result
    except Exception as e:
        print(f"Error in face_analysis: {e}")


if __name__ == '__main__':
    print(face_analysis(img='faces/img.png'))