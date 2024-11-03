import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load and display a 12-bit RAW Bayer image
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to load image at path '{path}'. Check the file path.")
    return image

# Demosaic function (only if the image is Bayer RAW)
def demosaic(image):
    if len(image.shape) == 2:
        print("Applying demosaicing on Bayer RAW image.")
        return cv2.cvtColor(image, cv2.COLOR_BayerGR2RGB)
    else:
        print("Image is already in RGB format; skipping demosaicing.")
        return image

# White Balance (Gray-World Algorithm)
def white_balance(img):
    r, g, b = cv2.split(img)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    avg_gray = (mean_r + mean_g + mean_b) / 3
    scale_r, scale_g, scale_b = avg_gray / mean_r, avg_gray / mean_g, avg_gray / mean_b
    r = cv2.multiply(r, scale_r)
    g = cv2.multiply(g, scale_g)
    b = cv2.multiply(b, scale_b)
    return cv2.merge([r, g, b]).astype(np.uint8)

# Denoising with Gaussian Filter
def denoise(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Gamma Correction with Interactive Slider
def gamma_correction(image, gamma=2.2):
    image = image / 255.0
    image = np.power(image, 1.0 / gamma)
    return (image * 255).astype(np.uint8)

# Sharpening with Unsharp Mask
def unsharp_mask(image, strength=1.5):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

# Contrast Enhancement (CLAHE)
def contrast_enhancement(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# Main function to process the image
def main():
    image_path = r'C:\Users\dheem\OneDrive\Desktop\emmatra\dino2.jpg'
    image = load_image(image_path)
    
    if image is None:
        return
    
    rgb_image = demosaic(image)
    balanced_image = white_balance(rgb_image)
    enhanced_image = contrast_enhancement(balanced_image)  # New contrast enhancement
    denoised_image = denoise(enhanced_image)
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    img_display = ax.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))

    ax_gamma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    gamma_slider = Slider(ax_gamma, 'Gamma', 0.1, 3.0, valinit=2.2)

    def update(val):
        gamma_val = gamma_slider.val
        gamma_corrected_image = gamma_correction(denoised_image, gamma=gamma_val)
        sharpened_image = unsharp_mask(gamma_corrected_image)
        img_display.set_data(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

    gamma_slider.on_changed(update)
    
    plt.title("Enhanced Image with Interactive Gamma Correction")
    plt.show()

if __name__ == "__main__":
    main()
