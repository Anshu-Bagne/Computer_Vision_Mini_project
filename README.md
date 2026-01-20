````markdown
# Feature Detection & Matching with OpenCV (SIFT, Harris, RANSAC)

**Author:** Anshu Bagne  
**Tech Stack:** Python, OpenCV, NumPy, Matplotlib  

This repository demonstrates three important computer vision techniques using **OpenCV**:

1. **SIFT keypoint detection + FLANN matching** between two images  
2. **Harris corner detection** and visualization (including rotated images)  
3. **RANSAC-based outlier removal** for robust keypoint matching and transformation estimation  

---

## 📌 Requirements

Install the required libraries:

```bash
pip install opencv-python numpy matplotlib
````

✅ Example Output while installing OpenCV:

```bash
Requirement already satisfied: opencv-python ...
Requirement already satisfied: numpy>=1.21.2 ...
```

---

## 📂 Project Structure

Keep your project files like this:

```
.
├── sift_flann_match.py
├── harris_corner_detection.py
├── ransac_matching.py
├── bottle.webp
├── mug.webp
├── square_grid.jpg
└── README.md
```

---

## ✅ Task 1: SIFT Keypoint Detection + FLANN Feature Matching

### 🎯 Objective

Detect and match **keypoints** between:

* an original image
* a rotated + scaled version of the same image

### 🔑 Key Concepts

* **SIFT (Scale-Invariant Feature Transform)** finds stable keypoints across scale/rotation changes.
* **FLANN (Fast Library for Approximate Nearest Neighbors)** performs fast matching.
* **Lowe’s Ratio Test** helps remove weak/incorrect matches.

### 🧾 Code (`sift_flann_match.py`)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original image in grayscale
img1 = cv2.imread("bottle.webp", cv2.IMREAD_GRAYSCALE)

# Show original image
plt.figure(figsize=(6, 6))
plt.imshow(img1, cmap='gray')
plt.title("Original Image")
plt.axis("off")
plt.show()

# Generate rotated and scaled version
rows, cols = img1.shape
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
img2 = cv2.warpAffine(img1, M, (cols, rows))

# Detect keypoints and compute descriptors using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Use FLANN for feature matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to find good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw the good matches
result_img = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Show result
plt.figure(figsize=(16, 8))
plt.imshow(result_img)
plt.title("SIFT + FLANN Matches")
plt.axis("off")
plt.show()
```

### ✅ Output

A plot showing feature matches between the original and transformed image.

---

## ✅ Task 2: Harris Corner Detector (Corner Visualization)

### 🎯 Objective

Detect and display **corner points** in a grayscale image using the **Harris corner detection** method.

### 🔑 Key Concepts

* Corners are points with **high intensity change in both x and y directions**.
* `cv2.cornerHarris()` returns a corner response map.
* Applying a **threshold** identifies strong corners.

---

### 🧾 Code (`harris_corner_detection.py`)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("square_grid.jpg")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_copy = np.copy(image_rgb)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 3, 0.09)
dst = cv2.dilate(dst, None)

# Thresholding
thresh = 0.1 * dst.max()
corner_image = np.copy(image_copy)

for j in range(dst.shape[0]):
    for i in range(dst.shape[1]):
        if dst[j, i] > thresh:
            cv2.circle(corner_image, (i, j), 1, (255, 0, 0), -5)

plt.figure(figsize=(8, 6))
plt.imshow(corner_image)
plt.axis("off")
plt.title("Harris Corner Detection")
plt.show()
```

---

### 🔁 Harris Corners on Rotated Image (Rotation Test)

This checks whether corners are still detected after rotating the image.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("square_grid.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

(h, w) = image_rgb.shape[:2]
center = (w // 2, h // 2)
angle = 45

rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (w, h))

gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2GRAY)
corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=5, k=0.1)

thresh = 0.1 * corners.max()
corner_image = np.copy(rotated_image)

for j in range(corners.shape[0]):
    for i in range(corners.shape[1]):
        if corners[j, i] > thresh:
            cv2.circle(corner_image, (i, j), 3, (255, 0, 0), -5)

plt.figure(figsize=(8, 6))
plt.imshow(corner_image)
plt.axis("off")
plt.title(f"Harris Corner Detection on Rotated Image ({angle}°)")
plt.show()
```

---

## ✅ Task 3: RANSAC for Outlier Removal + Transformation Model Fitting

### 🎯 Objective

Use **RANSAC** to remove incorrect keypoint matches and estimate a robust transformation model between two images.

### 🔑 Key Concepts

* ORB features are fast and work well with binary descriptors.
* Many matches contain outliers → **RANSAC helps reject them**.
* `cv2.findHomography(..., cv2.RANSAC)` estimates a stable mapping between points.

---

### 🧾 Code (`ransac_matching.py`)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original image
img1 = cv2.imread("mug.webp", cv2.IMREAD_GRAYSCALE)

# Generate transformed image (rotate + scale)
rows, cols = img1.shape
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
img2 = cv2.warpAffine(img1, M, (cols, rows))

# Detect keypoints and descriptors using ORB
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match descriptors using Brute Force + Hamming
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# RANSAC - Estimate Homography & Mask Inliers
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

# Draw inlier matches
result_img = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,
    matchesMask=matches_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(result_img)
plt.title("RANSAC Matching Between Original and Transformed Image")
plt.axis("off")
plt.show()
```

### ✅ Output

A visualization of good matches where **RANSAC inliers are highlighted**, giving a more accurate match set.

---

## ▶️ How to Run

Run each script separately:

```bash
python sift_flann_match.py
python harris_corner_detection.py
python ransac_matching.py
```

---

## 📌 Notes / Troubleshooting

### ⚠️ If images do not load:

Make sure the image filenames and paths match your folder exactly:

```python
cv2.imread("bottle.webp")
cv2.imread("mug.webp")
cv2.imread("square_grid.jpg")
```

### ✅ Recommended Fix (Best Practice)

Always check if the image is loaded:

```python
if img1 is None:
    raise FileNotFoundError("Image not found! Check filename/path.")
```

---

## 📌 Summary

| Technique              | Purpose                             | Output                           |
| ---------------------- | ----------------------------------- | -------------------------------- |
| SIFT + FLANN           | Feature detection and matching      | Matched keypoints visualization  |
| Harris Corner Detector | Corner detection                    | Corners plotted on image         |
| ORB + RANSAC           | Robust matching + outlier rejection | Clean inlier match visualization |

---

## 📜 License

This project is for educational purposes. You may reuse and modify it freely.

---

```
```
