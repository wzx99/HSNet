# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import interpolate
from scipy.special import comb as n_over_k
import glob, os
import cv2

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import matplotlib.pyplot as plt
import math
import numpy as np
import random
# from scipy.optimize import leastsq
import torch
from torch import nn
from torch.nn import functional as F

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

from shapely.geometry import *
from PIL import Image
import time

import re 
import xml.etree.cElementTree as ET


label_path = 'path/to/totaltext/train_gts/'

class Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.x0 = ps[0, 0]
        self.x3 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y3 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 81))

    def forward(self):
        x0, x1, x2, x3, y0, y1, y2, y3 = self.control_points()
        t = self.t
        bezier_x = (1-t)*((1-t)*((1-t)*x0+t*x1)+t*((1-t)*x1+t*x2))+t*((1-t)*((1-t)*x1+t*x2)+t*((1-t)*x2+t*x3))
        bezier_y = (1-t)*((1-t)*((1-t)*y0+t*y1)+t*((1-t)*y1+t*y2))+t*((1-t)*((1-t)*y1+t*y2)+t*((1-t)*y2+t*y3))
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.y0, self.y1, self.y2, self.y3

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3, self.y0, self.y1.item(), self.y2.item(), self.y3


def train(x, y, ctps, lr):
    x, y = np.array(x), np.array(y)
    ps = np.vstack((x, y)).transpose()
    bezier = Bezier(ps, ctps)
    
    return bezier.control_points_f()

def draw(ps, control_points, t):
    x = ps[:, 0]
    y = ps[:, 1]
    x0, x1, x2, x3, y0, y1, y2, y3 = control_points
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,color='m',linestyle='',marker='.')
    bezier_x = (1-t)*((1-t)*((1-t)*x0+t*x1)+t*((1-t)*x1+t*x2))+t*((1-t)*((1-t)*x1+t*x2)+t*((1-t)*x2+t*x3))
    bezier_y = (1-t)*((1-t)*((1-t)*y0+t*y1)+t*((1-t)*y1+t*y2))+t*((1-t)*((1-t)*y1+t*y2)+t*((1-t)*y2+t*y3))

    plt.plot(bezier_x,bezier_y, 'g-')
    plt.draw()
    plt.pause(1) # <-------
    raw_input("<Hit Enter To Close>")
    plt.close(fig)


Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * n_over_k(n,k)
BezierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]


def bezier_fit(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))
    Pseudoinverse = np.linalg.pinv(BezierCoeff(t)) # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(data)     # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1,:].flatten().tolist()
    return medi_ctp

def bezier_fitv2(x, y):
    xc01 = (2*x[0] + x[-1])/3.0
    yc01 = (2*y[0] + y[-1])/3.0
    xc02 = (x[0] + 2* x[-1])/3.0
    yc02 = (y[0] + 2* y[-1])/3.0
    control_points = [xc01,yc01,xc02,yc02]
    return control_points

def is_close_to_line(xs, ys, thres):
        regression_model = LinearRegression()
        # Fit the data(train the model)
        regression_model.fit(xs.reshape(-1,1), ys.reshape(-1,1))
        # Predict
        y_predicted = regression_model.predict(xs.reshape(-1,1))

        # model evaluation
        rmse = mean_squared_error(ys.reshape(-1,1)**2, y_predicted**2)
        rmse = rmse/(ys.reshape(-1,1)**2- y_predicted**2).max()**2

        if rmse > thres:
                return 0.0
        else:
                return 2.0

def is_close_to_linev2(xs, ys, size, thres = 0.05):
        pts = []
        nor_pixel = int(size**0.5)
        for i in range(len(xs)):
                pts.append(Point([xs[i], ys[i]]))
        import itertools
        # iterate by pairs of points
        slopes = [(second.y-first.y)/(second.x-first.x) if not (second.x-first.x) == 0.0 else math.inf*np.sign((second.y-first.y)) for first, second in zip(pts, pts[1:])]
        st_slope = (ys[-1] - ys[0])/(xs[-1] - xs[0])
        max_dis = ((ys[-1] - ys[0])**2 +(xs[-1] - xs[0])**2)**(0.5)

        diffs = abs(slopes - st_slope)
        score = diffs.sum() * max_dis/nor_pixel

        if score < thres:
                return 0.0
        else:
                return 3.0

# test
labels = glob.glob(label_path+'*.txt')
labels.sort()

if not os.path.exists(label_path.replace('train_gts/', 'bezier_train_gts/')):
    os.mkdir(label_path.replace('train_gts/', 'bezier_train_gts/'))

ignore = 0
total = 0
ignore2 = 0
ignore_total = 0
for il, label in enumerate(labels):
    print('Processing: '+label)
    imgdir = label.replace('train_gts/', 'train_images/').replace('.txt', '')

    outgt = open(label.replace('train_gts/', 'bezier_train_gts/'), 'w', encoding="utf-8")

    data = []
    cts  = []
    polys = []
        
    fin = open(label, 'r').readlines()
    for il, line in enumerate(fin):
        line = line.strip().split(',')
        if (not len(line[:-1])%2==0) or len(line[:-1])<8: 
            # print(111, line[:-1])
            total += 1
            ignore += 1
            continue
        elif len(line[:-1])==12:
            line = [
                line[0],line[1],
                str((float(line[0])+float(line[2]))/2), str((float(line[1])+float(line[3]))/2),
                line[2],line[3],
                line[2],line[3],
                str((float(line[2])+float(line[4]))/2), str((float(line[3])+float(line[5]))/2),
                line[4],line[5],
                line[6],line[7],
                str((float(line[6])+float(line[8]))/2), str((float(line[7])+float(line[9]))/2),
                line[8],line[9],
                line[8],line[9],
                str((float(line[8])+float(line[10]))/2), str((float(line[9])+float(line[11]))/2),
                line[10],line[11],
                line[12],
            ]
        elif len(line[:-1])==8:
            line = [
                line[0],line[1],
                str((3*float(line[0])+float(line[2]))/4), str((3*float(line[1])+float(line[3]))/4),
                str((float(line[0])+float(line[2]))/2), str((float(line[1])+float(line[3]))/2),
                str((float(line[0])+float(line[2]))/2), str((float(line[1])+float(line[3]))/2),
                str((float(line[0])+3*float(line[2]))/4), str((float(line[1])+3*float(line[3]))/4),
                line[2],line[3],
                line[4],line[5],
                str((3*float(line[4])+float(line[6]))/4), str((3*float(line[5])+float(line[7]))/4),
                str((float(line[4])+float(line[6]))/2), str((float(line[5])+float(line[7]))/2),
                str((float(line[4])+float(line[6]))/2), str((float(line[5])+float(line[7]))/2),         
                str((float(line[4])+3*float(line[6]))/4), str((float(line[5])+3*float(line[7]))/4),         
                line[6],line[7],
                line[8],
            ]
        ct = line[-1]
        if ct == '###': 
            ignore_total += 1
        coords = [(float(line[:-1][ix]), float(line[:-1][ix+1])) for ix in range(0, len(line[:-1]), 2)]
        poly = Polygon(coords)
        data.append(np.array([float(x) for x in line[:-1]]))
        cts.append(ct)
        polys.append(poly)

    ############## top
    img = plt.imread(imgdir)
    
    for iid, ddata in enumerate(data):
        lh = len(data[iid])
#         assert(lh % 4 ==0)
        total += 1
        if lh % 4 != 0:
            ignore += 1
            # print(222, lh)
            continue
        lhc2 = int(lh/2)
        lhc4 = int(lh/4)
        xcors = [data[iid][i] for i in range(0, len(data[iid]),2)]
        ycors = [data[iid][i+1] for i in range(0, len(data[iid]),2)]

        curve_data_top = data[iid][0:lhc2].reshape(lhc4, 2)
        curve_data_bottom = data[iid][lhc2:].reshape(lhc4, 2)

        left_vertex_x = [curve_data_top[0,0], curve_data_bottom[lhc4-1,0]]
        left_vertex_y = [curve_data_top[0,1], curve_data_bottom[lhc4-1,1]]
        right_vertex_x = [curve_data_top[lhc4-1,0], curve_data_bottom[0,0]]
        right_vertex_y = [curve_data_top[lhc4-1,1], curve_data_bottom[0,1]]

        x_data = curve_data_top[:, 0]
        y_data = curve_data_top[:, 1]

        init_control_points = bezier_fit(x_data, y_data)

        learning_rate = is_close_to_linev2(x_data, y_data, img.size)

        x0, x1, x2, x3, y0, y1, y2, y3 = train(x_data, y_data, init_control_points, learning_rate)
        control_points = np.array([
                [x0,y0],\
                [x1,y1],\
                [x2,y2],\
                [x3,y3]                        
        ])

        x_data_b = curve_data_bottom[:, 0]
        y_data_b = curve_data_bottom[:, 1]

        init_control_points_b = bezier_fit(x_data_b, y_data_b)

        learning_rate = is_close_to_linev2(x_data_b, y_data_b, img.size)

        x0_b, x1_b, x2_b, x3_b, y0_b, y1_b, y2_b, y3_b = train(x_data_b, y_data_b, init_control_points_b, learning_rate)
        control_points_b = np.array([
                [x0_b,y0_b],\
                [x1_b,y1_b],\
                [x2_b,y2_b],\
                [x3_b,y3_b]                        
        ])


        outstr = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},####{}\n'.format(x0,y0,\
                                                                            round(x1, 2), round(y1, 2),\
                                                                            round(x2, 2), round(y2, 2),\
                                                                            round(x3, 2), round(y3, 2),\
                                                                            round(x0_b, 2), round(y0_b, 2),\
                                                                            round(x1_b, 2), round(y1_b, 2),\
                                                                            round(x2_b, 2), round(y2_b, 2),\
                                                                            round(x3_b, 2), round(y3_b, 2),\
                                                                            cts[iid])
        outgt.writelines(outstr)
    outgt.close()
print(ignore, ignore2, total, ignore_total)

