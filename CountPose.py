# code from https://gluon-cv.mxnet.io/build/examples_pose/cam_demo.html

from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints


class model:
    def __init__(self):
        # load detector
        self.ctx = mx.cpu()
        detector_name = "ssd_512_mobilenet1.0_coco"
        self.detector = get_model(detector_name, pretrained=True, ctx=ctx)
        self.detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
        self.detector.hybridize()
        self.estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
        self.estimator.hybridize()
        self.trajectory = {'palvic': [], 'knee': [], 'shoulders': []}
        self.mark_point = {'x': [], 'y': []}

    def detect_main_point(self, capture):
        #axes = None
        #num_frames = len(F)
        #for i in range(num_frames):
        while True:
            ret, frame = capture.read()
            if ret is None:
                break
            #frame = F[i]
            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
            x = x.as_in_context(self.ctx)
            class_IDs, scores, bounding_boxs = self.detector(x)

            pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                               output_shape=(128, 96), ctx=ctx)
            if len(upscale_bbox) > 0:
                predicted_heatmap = self.estimator(pose_input)
                pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

                #img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                #                        box_thresh=0.5, keypoint_thresh=0.2)
                # mark important point of body

                # img = cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
                # mark knee and sholders

                # img = cv2.circle(img, kn, 4, (0, 255, 0), -1)
                # img = cv2.circle(img, shl, 4, (0, 255, 0), -1)
                _, plv = pred_coords.asnumpy()[:, 11][0]
                _, shl = pred_coords.asnumpy()[:, 5][0]
                _, kn = pred_coords.asnumpy()[:, 14][0]
                self.trajectory['palvic'].append(plv)
                self.trajectory['knee'].append(kn)
                self.trajectory['shoulders'].append(shl)
            #try:
                #pass
                # print('img')
                ### not work at colab#cv_plot_image(img)
                # cv2_imshow(img)
                # cv2.waitKey(1)
            #except Exception as e:
            #    print(e)
    def __count_points(self):
        pavl, knee, shld = self.trajectory.values()
        n = len(pavl)
        high = min(knee[n // 4: n * 3 // 4])
        low = max(shld[n // 4: n * 3 // 4])
        max_high = max(knee[n // 4: n * 3 // 4]) + (high - low)
        count = 0
        stand = True
        #print(high, low, max_high)
        for i, pvl_y in enumerate(pavl):
            if stand and (pvl_y > high) and (pvl_y < max_high):
                self.mark_point['x'].append(i)
                self.mark_point['y'].append(pvl_y)
                count += 1
                stand = False
            elif not stand and (pvl_y < low):
                stand = True
        return count#, self.mark_point

    def count_squards(self, capture):
        self.detect_main_point(capture)
        return self.__count_points()[0]
