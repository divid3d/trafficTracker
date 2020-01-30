import cv2

path = "C://Users//Divided//Desktop//klatki//"
filename = "C://Users//Divided//Desktop//klatki//1.jpg"
img = cv2.imread(filename)
height, width, layers = img.shape
size = (width, height)
out = cv2.VideoWriter(path + 'test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for count in range(len(cv2.os.listdir(path))):
    filename = "C://Users//Divided//Desktop//klatki//" + str(count) + '.jpg'
    img = cv2.imread(filename)
    out.write(img)
    print("appended: " + str(count) + " out of " + str(len(cv2.os.listdir(path))))

out.release()
