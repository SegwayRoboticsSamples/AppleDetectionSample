import numpy as np
def check_duplicate_weights(model):
    l = [x.name for x in model.weights]
    duplicates = set([x for x in l if l.count(x) > 1])
    if len(duplicates) > 1:
        print('duplicate tensors:', duplicates)
        exit()

def nms(boxes, number_classes, overlap_threshold=0.3, has_class_id=False):
    if len(boxes) == 0:
        nms_boxes = []
    else:
        nms_boxes = []
        for k in range(number_classes):
            if has_class_id:
                class_id = boxes[:, 6]
                boxes_same_class = boxes[class_id ==k]
                score = boxes_same_class[:, 5]
                x1 = boxes_same_class[:, 0]
                y1 = boxes_same_class[:, 1]
                x2 = boxes_same_class[:, 2]
                y2 = boxes_same_class[:, 3]
                image_ids = boxes_same_class[:, 4]
                
            else:
                score = boxes[:, 5]
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
                image_ids = boxes[:, 4]
            area = (x2 - x1 + 0.0001) * (y2 - y1 + 0.0001)

            #vals = sort(score)
            I = np.argsort(score)
            pick = []
            while (I.size != 0):
                last = I.size
                i = I[last - 1]
                pick.append(i)
                suppress = [last - 1]
                for pos in range(last - 1):
                    j = I[pos]
                    xx1 = max(x1[i], x1[j])
                    yy1 = max(y1[i], y1[j])
                    xx2 = min(x2[i], x2[j])
                    yy2 = min(y2[i], y2[j])
                    w = xx2 - xx1 + 0.0001
                    h = yy2 - yy1 + 0.0001
                    if (w > 0 and h > 0):
                        o = w * h / area[j]
                        if (o > overlap_threshold):
                            suppress.append(pos)
                I = np.delete(I, suppress)
            for pick_id in pick:
                nms_boxes.append([x1[pick_id], y1[pick_id], x2[pick_id],
                y2[pick_id], k, score[pick_id], image_ids[pick_id]])
    return nms_boxes

def interset(box1, box2):
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    return [left, top, right, bottom]

def overlap(bbx1, bbx2):
    left = max(-bbx1[2] / 2.0, -bbx2[2] / 2.0)
    top = max(-bbx1[3] / 2.0, -bbx2[3] / 2.0)
    right = min(bbx1[2] / 2.0, bbx2[2] / 2.0)
    bottom = min(bbx1[3] / 2.0, bbx2[3] / 2.0)
    if right - left < 0:
        return 0
    if bottom - top < 0:
        return 0
    intersect_area = (right - left) * (bottom - top)
    union_area = ((bbx1[2] - bbx1[0]) * (bbx1[3] - bbx1[1]) + 
        (bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1]) - intersect_area)

    return intersect_area / union_area
