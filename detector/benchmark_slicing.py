from PIL import Image
import time
import cv2
import torch
import torchvision

img_path = "/home/rohan/Documents/Uni/MA/slicing/img-1711300853.2299805.png"

img = cv2.imread(img_path)
img_pil = Image.open(img_path)
img_torch = torch.tensor(img)


target_gsd = 0.2
img_w, img_h = img.shape[1], img.shape[0]
altitude = 200
focal_len_mm = 3.04
sensor_width_mm = 3.68
sensor_width_px = 3280  # ToDo make changable
focal_len_px = focal_len_mm * (sensor_width_px / sensor_width_mm)
img_size_w = (altitude / focal_len_px * img_w) / target_gsd
img_size_h = (altitude / focal_len_px * img_h) / target_gsd

print(img_size_w, img_size_h)

scale_factor_w = img_size_w / img_w
scale_factor_h = img_size_h / img_h

start = time.perf_counter()
img = cv2.resize(img, (int(img_size_w), int(img_size_h)), interpolation=cv2.INTER_CUBIC)
end = time.perf_counter()
print("CV2 Time taken: ", end - start)

start = time.perf_counter()
img_pil = img_pil.resize((int(img_size_w), int(img_size_h)), Image.BICUBIC)
end = time.perf_counter()
print("PIL Time taken: ", end - start)

# start = time.perf_counter()
# img_torch = torch.nn.functional.interpolate(
#     img_torch.unsqueeze(0), (int(img_size_h), int(img_size_w)), mode="bicubic"
# )
# end = time.perf_counter()
# print("Torch interpolate Time taken: ", end - start)

# torch_resize = torchvision.transforms.Resize((int(img_size_h), int(img_size_w)))
# start = time.perf_counter()
# img_torch = torch_resize(img_torch)
# end = time.perf_counter()
# print("Torch Resize Time taken: ", end - start)
