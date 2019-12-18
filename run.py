import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

resizedX, resizedY = 416, 416
confidance = .50
inputs = tf.placeholder(tf.float32, [None, resizedX, resizedY, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)

# to display other detected #objects,change the classes and list of classes to their respective #COCO indices available in their website. Here 0th index is for #people and 1 for bicycle and so on. If you want to detect all the #classes, add the indices to this list
classes = {'2': 'car', '5': 'bus', '7': 'truck'}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
list_of_classes = [2, 5,
                   7]
with tf.Session() as sess:
    sess.run(model.pretrained())

    cap = cv2.VideoCapture("C://Users//Divided//Desktop//traffic_detection.mp4")
    # change the path to your directory or to '0' for webcam
    while cap.isOpened():
        ret, frame = cap.read()
        frameHeight, frameWidth = frame.shape[:2]
        scaleFactorX, scaleFactorY = frameWidth / resizedX, frameHeight / resizedY

        img = cv2.resize(frame, (resizedY, resizedX))

        imge = np.array(img).reshape(-1, resizedY, resizedX, 3)
        start_time = time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

        fps = 1 / (time.time() - start_time)
        print("FPS: %.2f" % fps)  # to time it
        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('image', frameWidth, frameHeight)

        boxes1 = np.array(boxes)
        for j in list_of_classes:  # iterate over classes
            count = 0
            if str(j) in classes:
                lab = classes[str(j)]
            if len(boxes1) != 0:
                # iterate over detected vehicles
                for i in range(len(boxes1[j])):
                    box = boxes1[j][i]
                    # setting confidence threshold
                    if boxes1[j][i][4] >= confidance:
                        count += 1

                        box[0] = box[0] * scaleFactorX
                        box[1] = box[1] * scaleFactorY
                        box[2] = box[2] * scaleFactorX
                        box[3] = box[3] * scaleFactorY

                        classColor = (255, 255, 255)
                        if classes[str(j)] == 'car':
                            classColor = colors[0]
                        elif classes[str(j)] == 'bus':
                            classColor = colors[1]
                        else:
                            classColor = colors[2]

                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), classColor, 2)
                        boxCenterX, boxCenterY = box[0] + ((box[2] - box[0]) / 2), box[1] + ((box[3] - box[1]) / 2)
                        cv2.circle(frame, (int(boxCenterX), int(boxCenterY)), 3, classColor, -1)

                        cv2.putText(frame, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                    lineType=cv2.LINE_AA)
            print(lab, ": ", count)

        # Display the output
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
