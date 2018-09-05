# coding:utf-8
import sys
import numpy as np

sys.path.append("..")
import argparse
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
import cv2
import os


def read_gt_bbox(raw_list):
    list_len = len(raw_list)
    bbox_num = (list_len - 1) // 4
    idx = 1
    bboxes = np.zeros((bbox_num, 4), dtype=int)
    for i in range(4):
        for j in range(bbox_num):
            bboxes[j][i] = int(raw_list[idx])
            idx += 1
    return bboxes


def get_image_info(anno_file):
    f = open(anno_file, 'r')
    image_info = []
    for line in f:
        ct_list = line.strip().split(' ')
        path = ct_list[0]

        path_list = path.split('\\')
        event = path_list[0]
        name = path_list[1]
        # print(event, name )
        bboxes = read_gt_bbox(ct_list)
        image_info.append([event, name, bboxes])
    print('total number of images in validation set: ', len(image_info))
    return image_info


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='ONet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['../data/MTCNN_l1Smoothed_model/PNet_landmark/PNet',
                                 '../data/MTCNN_l1Smoothed_model/RNet_landmark/RNet',
                                 '../data/MTCNN_l1Smoothed_model/ONet_landmark/ONet'],
                        type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[18, 20, 18], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.6, 0.3, 0.1], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=12, type=int)
    parser.add_argument('--output_dir', dest='output_dir', help='output path of the evaluation result',
                        default='../../DATA/ONet/ONet-631-epoch18_mf12', type=str)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    # parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')

    args = parser.parse_args()
    return args


def run(prefix, epoch, batch_size, output_dir,
        test_mode="PNet",
        thresh=[0.3, 0.1, 0.7], min_face_size=20, stride=2, slide_window=False, shuffle=False, vis=False):
    data_dir = '../../DATA/WIDER_val/images'
    anno_file = 'wider_face_val.txt'
    output_file = output_dir
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    test_mode = test_mode
    thresh = thresh
    min_face_size = min_face_size
    stride = stride
    slide_window = slide_window
    shuffle = shuffle
    vis = vis
    detectors = [None, None, None]
    # prefix is the model path
    prefix = prefix
    epoch = epoch
    batch_size = batch_size
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    image_info = get_image_info(anno_file)

    current_event = ''
    save_path = ''
    idx = 0
    for item in image_info:
        idx += 1
        image_file_name = os.path.join(data_dir, item[0], item[1])
        if current_event != item[0]:

            current_event = item[0]
            save_path = os.path.join(output_file, item[0])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print('current path:', current_event)

        # generate detection
        img = cv2.imread(image_file_name)
        all_boxes, _ = mtcnn_detector.detect_single_image(img)

        f_name = item[1].split('.jpg')[0]

        dets_file_name = os.path.join(save_path, f_name + '.txt')
        fid = open(dets_file_name, 'w')
        boxes = all_boxes[0]
        if boxes is None:
            fid.write(item[1] + '\n')
            fid.write(str(1) + '\n')
            fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
            continue
        fid.write(item[1] + '\n')
        fid.write(str(len(boxes)) + '\n')

        for box in boxes:
            fid.write('%f %f %f %f %f\n' % (
                float(box[0]), float(box[1]), float(box[2] - box[0] + 1), float(box[3] - box[1] + 1), box[4]))

        fid.close()
        if idx % 10 == 0:
            print(idx)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args.prefix,
        args.epoch,
        args.batch_size,
        args.output_dir,
        args.test_mode,
        args.thresh,
        args.min_face,
        )