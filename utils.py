def resize_box(box, scaleFactorX, scaleFactorY):
    return [box[0] * scaleFactorX, box[1] * scaleFactorY, box[2] * scaleFactorX, box[3] * scaleFactorY]


def filter_box_by_size(boxes, minWidth=20, minHeight=20, minArea=600):
    if len(boxes) == 0:
        return []

    pick = []

    for i in range(len(boxes)):
        height = boxes[i, 3] - boxes[i, 1]
        if height < minHeight:
            continue

        width = boxes[i, 2] - boxes[i, 0]
        if width < minWidth:
            continue

        area = width * height
        if area < minArea:
            continue

        pick.append(i)

    return boxes[pick]
