from src.projection import load_ct, normalize_hu, project_ap, project_lat
import matplotlib.pyplot as plt

# Replace with a real CT path later
# For now we fake data to test the pipeline
import numpy as np
ct = np.random.randn(128, 256, 256).astype(np.float32)

ct = normalize_hu(ct)
ap = project_ap(ct)
lat = project_lat(ct)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("AP")
plt.imshow(ap, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("LAT")
plt.imshow(lat, cmap="gray")
plt.axis("off")

plt.show()
