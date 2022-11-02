import numpy as np

def overlap2(bbx1, bbx2):
    top = max(bbx1[0], bbx2[0])
    left = max(bbx1[1], bbx2[1])
    bottom = min(bbx1[2], bbx2[2])
    right = min(bbx1[3], bbx2[3])
    if right - left < 0:
        return 0
    if bottom - top <0:
        return 0
    intersect_area = (right - left) * (bottom - top)
    union_area = ((bbx1[2] - bbx1[0]) * (bbx1[3] - bbx1[1]) + 
        (bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1]) - intersect_area)
        
    return intersect_area / union_area
    
def calculate_averge_precision(predicted_boxs, groundtruth_dict):
    """ap of a single class """
    predicted_boxs.sort(key=lambda s: s[5], reverse=True)
    precision_recall = []
    TP = 0.0
    FP = 0.0
    positive_number = 0
    for k, v in groundtruth_dict.items():
        positive_number += len(v)
    if positive_number == 0:
        print("missing this class in test data")
        #return 0.0
    #print(predicted_boxs)
    predict_box={}
    for bbx in predicted_boxs:
        max_overlap = 0
        image_id = int(bbx[6])
        #print(bbx[0:7])
        if image_id in groundtruth_dict:
            for groundtruth in groundtruth_dict[image_id]:
                #print("ground",image_id, groundtruth)
                o = overlap2(groundtruth, bbx[0:4])
                if o > max_overlap:
                    max_overlap = o
        #print(image_id, bbx[0:4])
        if max_overlap > 0.5:
            TP = TP + 1
            #print(image_id,bbx[0:4])
            if image_id not in predict_box:
                predict_box[image_id]=[bbx[0:4]]
            else:
                predict_box[image_id].append(bbx[0:4])
        else:
            FP = FP + 1
            if image_id not in predict_box:
                predict_box[image_id]=[bbx[0:4]]
            else:
                predict_box[image_id].append(bbx[0:4])
        if positive_number > 0:
            precision_recall.append((TP / (TP + FP), TP / positive_number))

    print(TP,FP,positive_number)
    if positive_number > 0 and (TP + FP) > 0:
        print("precision: ", TP / (TP + FP))
        print("recall: ", TP / positive_number)
    #print(precision_recall[-1])
    ap = 0.0
    # old function
    # for t in range(0, 11, 1):
    #     precisions = [x[0] for x in precision_recall if x[1] > t / 10.0]
    #     if len(precisions) > 0:
    #         p = max(precisions)
    #     else:
    #         p = 0
    #     ap = ap + p / 11
    # new function
    rec =  [pr[1] for pr in precision_recall]
    prec = [pr[0] for pr in precision_recall]
    rec= np.array(rec)
    prec = np.array(prec)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # check bigger prec with equal rec.
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    
    # check the changed location of rec.
    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap,predict_box
