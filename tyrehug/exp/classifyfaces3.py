from settings import DATA_DIR
import cv2
import numpy
import os

image_path = os.path.join(DATA_DIR, "YALE", "centered")
image_filenames = [filename for filename in os.listdir(image_path)]
images = []
labels = []

for filename in image_filenames:
    print(filename)
    image = cv2.imread(os.path.join(image_path, filename), 0)
    images.append(image)

    label = int(filename[7:9])
    labels.append(label)

images = numpy.array(images)
print(labels)

# Copy classifyfaces2 approach, also try default keras models
# Compare with random forests
