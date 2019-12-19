import cv2

img_array = []
path = "C://Users//Divided//Desktop//klatki//"

for count in range(len(cv2.os.listdir(path))):
    filename = "C://Users//Divided//Desktop//klatki//" +str(count) + '.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

    out = cv2.VideoWriter(path + 'test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
        print(str(i) + '/' + str(range(len(img_array))))
    out.release()