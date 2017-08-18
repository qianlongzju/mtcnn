import numpy as np
import mxnet as mx
import os
import sys
import argparse
import cv2
import time
from core.symbol import P_Net, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector

nc_image_dir = "/Users/isaiah/Workspace/real_time_face_recognition/nc_face_images"
nc_image_dir = "/Users/isaiah/Workspace/real_time_face_recognition/nc_face_images_bak"
result_dir = "/Users/isaiah/Workspace/real_time_face_recognition/"

def test_net(prefix, epoch, batch_size, ctx,
             thresh=[0.6, 0.6, 0.7], min_face_size=24,
             stride=2, slide_window=False, photo='test01.jpg'):

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    if slide_window:
        PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
    else:
        PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    if os.path.isdir(photo):
        test_dir(mtcnn_detector, photo)
    else:
        test_photo(mtcnn_detector, photo)

def test_photo(detector, photo):
    img = cv2.imread(photo)
    t1 = time.time()

    boxes_c = get_boxes_c(detector, img)

    print 'time: ',time.time() - t1

    if boxes_c is not None:
        show_boxes_c(img, boxes_c)

def test_dir(detector, test_dir):
    t1 = time.time()
    count = 0
    face_count = 0
    for f in os.listdir(test_dir):
        count += 1
        f_name = os.path.join(test_dir, f)
        #print f_name
        img = cv2.imread(f_name)
        boxes_c = get_boxes_c(detector, img)
        if boxes_c is not None:
            face_count += 1
            draw = get_img_with_bbox(img, boxes_c)
            cv2.imwrite(result_dir + 'result/true_20170818/%s.jpg' % f, draw)
            #show_boxes_c(img, boxes_c)
            #break
        else:
            #cv2.imwrite(result_dir + 'result/false_20170818/%s.jpg' % f, img)
            print f
        #break
    print 'time: ',time.time() - t1
    print 'face/total: %s/%s' % (face_count, count)

#@profile
def get_boxes_c(detector, img):
    boxes, boxes_c = detector.detect_pnet(img)
    if boxes_c is None:
        return None
    boxes, boxes_c = detector.detect_rnet(img, boxes_c)
    if boxes_c is None:
        return None
    boxes, boxes_c = detector.detect_onet(img, boxes_c)
    return boxes_c

def get_img_with_bbox(img, boxes_c):
    draw = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in boxes_c:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
        cv2.putText(draw, '%.3f'%b[4], (int(b[0]), int(b[1])), font, 0.4, (255, 255, 255), 1)
    return draw

def show_boxes_c(img, boxes_c):
    draw = get_img_with_bbox(img, boxes_c)
    cv2.imshow("detection result", draw)
    cv2.waitKey(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--photo', dest='photo', help='photo name',
                        default=nc_image_dir, type=str)
                        #default='test01.jpg', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['model/pnet', 'model/rnet', 'model/onet'], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.5, 0.5, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=40, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = mx.gpu(args.gpu_id)
    if args.gpu_id == -1:
        ctx = mx.cpu(0)
    test_net(args.prefix, args.epoch, args.batch_size,
             ctx, args.thresh, args.min_face,
             args.stride, args.slide_window, args.photo)
