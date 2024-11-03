import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


image_paths = [
    r"C:\Users\dheem\OneDrive\Desktop\emmatra\shadow.jpg",  
    r"C:\Users\dheem\OneDrive\Desktop\emmatra\correctly_exposed.jpg",  
    r"C:\Users\dheem\OneDrive\Desktop\emmatra\overexposed.jpg"   
]


images = []
for path in image_paths:
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            raise ValueError(f"File found but could not be read by OpenCV: {path}")
    else:
        raise ValueError(f"File not found at path: {path}")


if len(images) != 3:
    raise ValueError("Failed to load all images. Check paths and file integrity.")


min_height = min(img.shape[0] for img in images)
min_width = min(img.shape[1] for img in images)
images = [cv2.resize(img, (min_width, min_height)) for img in images]


exposure_times = np.array([1/125.0, 1/30.0, 1/8.0], dtype=np.float32)


merge_debevec = cv2.createMergeDebevec()
hdr_image = merge_debevec.process(images, times=exposure_times)


tonemap_mantiuk = cv2.createTonemapMantiuk(gamma=1.1, scale=0.9, saturation=1.3)
ldr_image = tonemap_mantiuk.process(hdr_image)


ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)  
ldr_image = cv2.convertScaleAbs(ldr_image, alpha=1.2, beta=10)  


plt.figure(figsize=(16, 8))


plt.subplot(221)
plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
plt.title('Underexposed (Shadow)')
plt.axis('off')


plt.subplot(222)
plt.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
plt.title('Correctly Exposed (Normal Light)')
plt.axis('off')


plt.subplot(223)
plt.imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
plt.title('Overexposed (Bright Light)')
plt.axis('off')


plt.subplot(224)
plt.imshow(cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB))
plt.title('Enhanced HDR Merged Image')
plt.axis('off')

plt.tight_layout()
plt.show()


output_path = r"C:\Users\dheem\OneDrive\Desktop\emmatra\enhanced_output_hdr_image.jpg"
cv2.imwrite(output_path, ldr_image)
print(f"Enhanced HDR merged and tone mapped image saved at: {output_path}")
