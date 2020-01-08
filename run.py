import time

import cv2
import tensorflow as tf
import tensornets as nets

from sort import *
from utils import resize_box

resizedX, resizedY = 416, 416
confidance = .4
inputs = tf.placeholder(tf.float32, [None, resizedX, resizedY, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
frameCounter = 0

# to display other detected #objects,change the classes and list of classes to their respective #COCO indices available in their website. Here 0th index is for #people and 1 for bicycle and so on. If you want to detect all the #classes, add the indices to this list
classes = {'1': 'bicycle', '2': 'car', '3': 'motorcycle', '5': 'bus', '7': 'truck'}
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
list_of_classes = [1, 2, 3, 5, 7]

mot_tracker = Sort(max_age=6, min_hits=3)
with tf.Session() as sess:
    sess.run(model.pretrained())

    cap = cv2.VideoCapture("C://Users//Divided//Desktop//traffic.mp4")
    # change the path to your directory or to '0' for webcam
    while cap.isOpened():
        ret, frame = cap.read()
        start_time = time.time()
        frameHeight, frameWidth = frame.shape[:2]
        scaleFactorX, scaleFactorY = frameWidth / resizedX, frameHeight / resizedY

        img = cv2.resize(frame, (resizedY, resizedX))

        imge = np.array(img).reshape(-1, resizedY, resizedX, 3)
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('image', frameWidth, frameHeight)

        boxes1 = np.array(boxes)
        detections = []
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
                        detections.append(resize_box(box[:4], scaleFactorX, scaleFactorY))
            print(lab, ": ", count)
        mot_tracker.update(np.array(detections))
        trackerObjects = mot_tracker.trackers
        for t in trackerObjects:
            if len(t.centroidHistory) > 0:
                cv2.putText(frame, str(t.id), (int(t.history[-1][0][0]), int(t.history[-1][0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255),
                            lineType=cv2.LINE_AA)
                cv2.polylines(frame, [np.asarray(t.centroidHistory).astype(int).reshape((-1, 1, 2))
                                      ], False, t.color, 1, cv2.LINE_AA)

            if len(t.history) > 0:
                cv2.rectangle(frame, (int(t.history[-1][0][0]), int(t.history[-1][0][1])),
                              (int(t.history[-1][0][2]), int(t.history[-1][0][3])), t.color, 1)

        fps = 1 / (time.time() - start_time)
        print("FPS: %.2f" % fps)  # to time it
        # Display the output
        cv2.imshow("image", frame)

        # path = "C://Users//Divided//Desktop//klatki"
        # cv2.imwrite(cv2.os.path.join(path, str(frameCounter) + ".jpg"), frame)
        # frameCounter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
