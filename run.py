import tensorflow as tf
import tensornets as nets
import cv2
import time
from sort import *

resizedX, resizedY = 416, 416
confidance = .3
inputs = tf.placeholder(tf.float32, [None, resizedX, resizedY, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
frameCounter = 0

# to display other detected #objects,change the classes and list of classes to their respective #COCO indices available in their website. Here 0th index is for #people and 1 for bicycle and so on. If you want to detect all the #classes, add the indices to this list
classes = {'2': 'car', '5': 'bus', '7': 'truck'}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
list_of_classes = [2, 5,
                   7]
detections = []
mot_tracker = Sort(10,5)
with tf.Session() as sess:
    sess.run(model.pretrained())

    cap = cv2.VideoCapture("C://Users//Divided//Desktop//traffic_test.mp4")
    # change the path to your directory or to '0' for webcam
    while cap.isOpened():
        ret, frame = cap.read()
        frameHeight, frameWidth = frame.shape[:2]
        scaleFactorX, scaleFactorY = frameWidth / resizedX, frameHeight / resizedY

        img = cv2.resize(frame, (resizedY, resizedX))

        imge = np.array(img).reshape(-1, resizedY, resizedX, 3)
        start_time = time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('image', frameWidth, frameHeight)

        boxes1 = np.array(boxes)
        detections.clear()
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

                        detections.append([box[0], box[1], box[2], box[3]])

                        classColor = (255, 255, 255)
                        if classes[str(j)] == 'car':
                            classColor = colors[0]
                        elif classes[str(j)] == 'bus':
                            classColor = colors[1]
                        else:
                            classColor = colors[2]

                        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), classColor, 2)
                        # boxCenterX, boxCenterY = box[0] + ((box[2] - box[0]) / 2), box[1] + ((box[3] - box[1]) / 2)
                        # cv2.circle(frame, (int(boxCenterX), int(boxCenterY)), 3, classColor, -1)
                        #
                        # cv2.putText(frame, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        #             lineType=cv2.LINE_AA)
            print(lab, ": ", count)
            trackers = mot_tracker.update(np.array(detections))
            # for d in trackers:
            #     cv2.circle(frame, (int(d[0] + ((d[2] - d[0]) / 2)), int(d[1] + ((d[3] - d[1]) / 2))), 3, classColor, -1)
            #     print(str(d[4]))

            trackerObjects = mot_tracker.trackers
            for t in trackerObjects:
                history = t.history
                pts  = []
                for historyEntry in history:
                    # cv2.circle(frame, (
                    #     int(historyEntry[0][0] + ((historyEntry[0][2] - historyEntry[0][0]) / 2)),
                    #     int(historyEntry[0][1] + ((historyEntry[0][3] - historyEntry[0][1]) / 2))), 3, t.color,
                    #            -1)
                    pts.append([int(historyEntry[0][0] + ((historyEntry[0][2] - historyEntry[0][0]) / 2)),int(historyEntry[0][1] + ((historyEntry[0][3] - historyEntry[0][1]) / 2))])

                if len(history) > 0:
                    cv2.rectangle(frame, (int(history[-1][0][0]), int(history[-1][0][1])),
                                  (int(history[-1][0][2]), int(history[-1][0][3])), t.color, 1)
                    cv2.putText(frame, str(t.id), (int(history[-1][0][0]), int(history[-1][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                (255, 255, 255),
                                lineType=cv2.LINE_AA)
                    pts = np.asarray(pts).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, t.color, 1, cv2.LINE_AA)

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
