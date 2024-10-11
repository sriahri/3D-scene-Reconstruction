import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cv2
from PIL import Image
import numpy as np
# image = cv2.imread('/home/labmember/Desktop/project/hamlyn_data/hamlyn_data/rectified01/depth01/0000000000.png')
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # plt.imshow(image, cmap='plasma')

# image = Image.open('/home/labmember/Desktop/project/hamlyn_data/hamlyn_data/rectified01/depth01/0000000000.png')
# # cv2.imshow("image", image)
# print(image.size)
# print(image.format)
# print(image.mode)
# # cv2.waitKey(0)
 
# # # It is for removing/deleting created GUI window from screen
# # # and memory
# # cv2.destroyAllWindows()

# image_data = np.array(image.getdata()).reshape(image.size[::-1])
# print(image_data)
# new_image = Image.fromarray(image_data)
# new_image.save('new_ground_depth.jpg')


image = mimg.imread('/home/labmember/Desktop/project/hamlyn_data/hamlyn_data/rectified01/depth01/0000000000.png')
image = image.astype(np.float32)/np.iinfo(np.uint16).max

print(image.shape)

plt.imshow(image, cmap='plasma')
plt.axis('off')
plt.show()