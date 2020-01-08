def resize_box(box, scaleFactorX, scaleFactorY):
    return [box[0] * scaleFactorX, box[1] * scaleFactorY, box[2] * scaleFactorX, box[3] * scaleFactorY]
