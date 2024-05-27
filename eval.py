import argparse
import csv
from collections import defaultdict
import os
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.linalg import sqrtm
import torch
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import lpips
from ultralytics import YOLO

# Check for GPU availability and configure settings
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available, using CPU instead.")

# Function to get image paths (not defined in the provided code)
def get_image_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Define ModelEvaluator class
class ModelEvaluator:
    def __init__(self, result_path):
        self.result_path = result_path
        self.topk = 1
        self.match_weight = 0.25
        self.method = 'vit'
        self.algo = 'max'
        self.batch_size = 64
        self.num_workers = 0

        # Initialize models
        self.inception_model = self.initialize_inception_model()
        self.lpips_net = self.get_lpips_net('cuda')
        self.face_detector = YOLO('./models/checkpoints/face_yolov8m.pt')

    def detect_and_crop_faces(self, image_pil):
        with torch.no_grad():
            results = self.face_detector(image_pil)
        if results and results[0].boxes:
            box = results[0].boxes.xyxy[0]
            x1, y1, x2, y2 = box[0].cpu().item(), box[1].cpu().item(), box[2].cpu().item(), box[3].cpu().item()
            if x2 > x1 and y2 > y1:
                cropped_face = image_pil.crop((x1, y1, x2, y2))
                return cropped_face
        return None

    def load_and_preprocess_image(self, image_path, target_size):
        image_pil = Image.open(image_path).convert('RGB')
        cropped_face = self.detect_and_crop_faces(image_pil)
        if cropped_face is None:
            raise Exception("No face detected in the image")
        cropped_face = cropped_face.resize(target_size)
        image = img_to_array(cropped_face)
        image = np.expand_dims(image, axis=0)
        image = np.clip(image, 0, 255).astype('uint8')
        return image

    def calculate_ssim(self, image1, image2):
        image1 = image1.squeeze()
        image2 = image2.squeeze()
        return ssim(image1, image2, channel_axis=-1, data_range=image2.max() - image2.min())

    def calculate_psnr(self, image1, image2):
        image1 = image1.squeeze()
        image2 = image2.squeeze()
        return psnr(image1, image2, data_range=image2.max() - image2.min())

    def calculate_fid(self, model, images1, images2):
        act1 = model.predict(preprocess_input(images1.copy()))
        act2 = model.predict(preprocess_input(images2.copy()))
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2)**2.0)
        try:
            covmean = sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        except ValueError:
            fid = float('nan')
        return fid

    def calculate_lpips(self, img1, img2, net):
        device = next(net.parameters()).device
        img1 = preprocess_input(img1.squeeze())
        img2 = preprocess_input(img2.squeeze())
        img1 = np.clip(img1, -1, 1)
        img2 = np.clip(img2, -1, 1)
        tensor_img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor_img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).to(device)
        return net(tensor_img1, tensor_img2).item()

    def get_lpips_net(self, device='cuda'):
        lpips_net = lpips.LPIPS(net='alex')
        lpips_net.to(device)
        return lpips_net

    def initialize_inception_model(self):
        model = InceptionV3(include_top=False, pooling='avg')
        return model

    def evaluate_images(self, image_path1, image_path2):
        target_size = (299, 299)
        img1 = self.load_and_preprocess_image(image_path1, target_size)
        img2 = self.load_and_preprocess_image(image_path2, target_size)

        ssim_value = self.calculate_ssim(img1[0], img2[0])
        psnr_value = self.calculate_psnr(img1[0], img2[0])
        fid_value = self.calculate_fid(self.inception_model, img1, img2)
        lpips_value = self.calculate_lpips(img1, img2, self.lpips_net)

        results = {
            "SSIM": ssim_value,
            "PSNR": psnr_value,
            "FID": fid_value,
            "LPIPS": lpips_value
        }

        with open(f"{self.result_path}/evaluation_results_{os.path.basename(image_path1).split('.')[0]}.txt", "a") as file:
            file.write("Evaluation Results:\n")
            file.write(f"{os.path.basename(image_path1)}\n")
            for key, value in results.items():
                file.write(f"{key}: {value}\n")

        return results

    def perform_evaluation(self, validation_image_paths, model_generation_image_paths):
        evaluation_results = []
        results_by_validation_paths = defaultdict(list)

        # Evaluate all combinations
        for validation_name, image_path2 in validation_image_paths.items():
            for model_category, model_paths_dict in model_generation_image_paths.items():
                for model_path_name, model_paths in model_paths_dict.items():
                    for image_path1 in model_paths:
                        results = self.evaluate_images(image_path1, image_path2)
                        evaluation_results.append({
                            'Validation Path': image_path1,
                            'Model Generation Path': image_path2,
                            'Evaluation Result': results
                        })
                        results_by_validation_paths[model_path_name].append(results)

        # Calculate and add average values for each validation path
        average_results = []
        for validation_name, results in results_by_validation_paths.items():
            avg_ssim = np.mean([r['SSIM'] for r in results])
            avg_psnr = np.mean([r['PSNR'] for r in results])
            avg_fid = np.mean([r['FID'] for r in results])
            avg_lpips = np.mean([r['LPIPS'] for r in results])

            average_results.append({
                'Validation Paths': validation_name,
                'Average SSIM': avg_ssim,
                'Average PSNR': avg_psnr,
                'Average FID': avg_fid,
                'Average LPIPS': avg_lpips
            })

        self.save_evaluation_results(evaluation_results, 'evaluation_results.csv')
        self.save_average_results(average_results, 'average_evaluation_results.csv')

    def save_evaluation_results(self, results, filename):
        fieldnames = ['Validation Path', 'Model Generation Path', 'Evaluation Result']

        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def save_average_results(self, results, filename):
        fieldnames = ['Validation Paths', 'Average SSIM', 'Average PSNR', 'Average FID', 'Average LPIPS']

        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def run(self, gt_images_path, validation_images_path):
        validation_image_paths = {
            'PEB_img_paths': get_image_paths(os.path.join(gt_images_path, 'PEB')),
            'PSJ_img_paths': get_image_paths(os.path.join(gt_images_path, 'PSJ')),
            'PSI_img_paths': get_image_paths(os.path.join(gt_images_path, 'PSI'))
        }

        model_generation_image_paths = {
            'FM': {
                'FM_PEB_img_paths': get_image_paths(os.path.join(validation_images_path, 'FM/PEB')),
                'FM_PSJ_img_paths': get_image_paths(os.path.join(validation_images_path, 'FM/PSJ')),
                'FM_PSI_img_paths': get_image_paths(os.path.join(validation_images_path, 'FM/PSI'))
            },
            'IPA': {
                'IPA_PEB_img_paths': get_image_paths(os.path.join(validation_images_path, 'IPA/PEB')),
                'IPA_PSJ_img_paths': get_image_paths(os.path.join(validation_images_path, 'IPA/PSJ')),
                'IPA_PSI_img_paths': get_image_paths(os.path.join(validation_images_path, 'IPA/PSI'))
            },
            'IPA_FM': {
                'IPA_FM_img_paths': get_image_paths(os.path.join(validation_images_path, 'IPA_FM'))
            },
            'comparison': {
                'comparison_img_paths': get_image_paths(os.path.join(validation_images_path, 'comparison/img')),
                'comparison_vid_paths': get_image_paths(os.path.join(validation_images_path, 'comparison/vid'))
            }
        }

        self.perform_evaluation(validation_image_paths, model_generation_image_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model generated images against ground truth images.")
    parser.add_argument('--gt_images_path', type=str, required=True, help='Path to the ground truth images.')
    parser.add_argument('--validation_images_path', type=str, required=True, help='Path to the validation images.')
    parser.add_argument('--result_path', type=str, required=True, help='Path to save the evaluation results.')

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.result_path)
    evaluator.run(args.gt_images_path, args.validation_images_path)