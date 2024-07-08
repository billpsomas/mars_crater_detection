import sys
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(dt_path):
    # load detection results, can be either a json file or a Python dictionary
    if isinstance(dt_path, str):
        dt = json.load(open(dt_path))
    else:
        dt = dt_path
    
    print(type(dt))
    assert type(dt) == dict, 'Detection result format is not correct.'
    
    ids = list(dt.keys())
    ids = [int(x) for x in ids]

    # convert result into COCO format
    dt_coco = []
    for idx, bboxes in dt.items():
        for box in bboxes:
            if not len(box) == 5:
                print(box)
                raise Exception("Prediction format should be [xmin, ymin, width, height, score].")
            dt_dic = {"image_id": int(idx), "category_id": 1, "bbox": box[:4], "score": box[-1]}
            dt_coco.append(dt_dic)

    # evaluation
    #dt_coco = dt
    ann_type = "bbox"
    coco_gt = COCO('gt_eval.json')
    coco_dt = coco_gt.loadRes(dt_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.maxDets = [1, 10,100]
    coco_eval.params.imgIds = ids
    coco_eval.params.iouThrs = [0.5]#[0.3, 0.4, 0.5, 0.6]
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 10.5 ** 2], [10.5 ** 2, 50.5 ** 2], [50.5 ** 2, 1e5 ** 2]]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


if __name__ == "__main__":
    if len(sys.argv) - 1 == 1:
        dt_path = sys.argv[1]
        evaluate(dt_path)
    else:
        raise ValueError('Expected 1 arguments: the detection results.')
