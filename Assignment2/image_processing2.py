import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


image_path = r"C:\Users\dheem\OneDrive\Desktop\emmatra\night.jpg" 
image = cv2.imread(image_path)


if image is None:
    raise ValueError("Image not found. Please check the image path.")




def white_balance(image):
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    avg = (avg_b + avg_g + avg_r) / 3
    
    
    image[:, :, 0] = np.clip(image[:, :, 0] * (avg / avg_b), 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * (avg / avg_g), 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * (avg / avg_r), 0, 255)
    
    return image.astype(np.uint8)


def denoise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def gamma_correction(image, gamma=2.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def sharpen(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

balanced_image = white_balance(image)
denoised_image = denoise(balanced_image)
gamma_corrected_image = gamma_correction(denoised_image)
sharpened_image = sharpen(gamma_corrected_image)


plt.figure(figsize=(12, 10))
plt.subplot(231), plt.imshow(cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB)), plt.title('White Balanced')
plt.axis('off')
plt.subplot(232), plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)), plt.title('Denoised')
plt.axis('off')
plt.subplot(233), plt.imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB)), plt.title('Gamma Corrected')
plt.axis('off')
plt.subplot(234), plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)), plt.title('Sharpened')
plt.axis('off')
plt.tight_layout()
plt.show()




def median_filter(image):
    return cv2.medianBlur(image, 5)

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def ai_denoise(image_path, model_path):
    
    model = tf.keras.models.load_model(model_path)
    
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    
    
    denoised_image = model.predict(img)
    denoised_image = np.squeeze(denoised_image) * 255.0
    
    return np.clip(denoised_image, 0, 255).astype(np.uint8)


def laplacian_filter(image):
    laplacian_img = cv2.Laplacian(image, cv2.CV_64F)
    
    
    laplacian_img_display = cv2.convertScaleAbs(laplacian_img)
    
    return laplacian_img_display


def compute_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    
    if noise == 0:
        return float('inf') 
    
    return 20 * np.log10(signal / noise)


median_filtered_image = median_filter(sharpened_image)
bilateral_filtered_image = bilateral_filter(sharpened_image)


ai_model_path = r"C:\Users\dheem\Downloads\gemma-2-2b-jpn-it-pytorch-gemma-2-2b-jpn-it-v1\model.ckpt"
ai_denoised_image = ai_denoise(image_path, ai_model_path)


laplacian_filtered_image = laplacian_filter(sharpened_image)


snr_median = compute_snr(median_filtered_image)
snr_bilateral = compute_snr(bilateral_filtered_image)
snr_ai = compute_snr(ai_denoised_image)


plt.figure(figsize=(12, 10))
plt.subplot(231), plt.imshow(cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Median Filtered')
plt.axis('off')
plt.subplot(232), plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Bilateral Filtered')
plt.axis('off')
plt.subplot(233), plt.imshow(cv2.cvtColor(ai_denoised_image, cv2.COLOR_BGR2RGB)), plt.title('AI Denoised')
plt.axis('off')
plt.subplot(234), plt.imshow(laplacian_filtered_image.astype(np.uint8), cmap='gray'), plt.title('Laplacian Filtered')
plt.axis('off')
plt.tight_layout()
plt.show()


print("\n--- Signal-to-Noise Ratio (SNR) Analysis ---")
print(f"SNR (Median Filter): {snr_median:.2f} dB")
print(f"SNR (Bilateral Filter): {snr_bilateral:.2f} dB")
print(f"SNR (AI Denoised): {snr_ai:.2f} dB")