import cv2
import os

def convert(img):
    if len(img.shape) > 2 and img.shape[2] == 4:
    #convert the image from RGBA2RGB
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
    

def main():
    path = "/home/labmember/Desktop/project/dataset/test/images/image01/"
    for name in os.listdir(path):
        if "Left" in name:
            new_path = path + name
            image = cv2.imread(new_path)
            new_image = convert(image)
            cv2.imwrite(new_path, new_image)

main()
