import cv2 as cv
import numpy as np
import os

"""
Link to original paper : https://arxiv.org/abs/1812.11703
Link to original repo  : https://github.com/STVIR/pysot

You can download the pre-trained weights of the Tracker Model from https://drive.google.com/file/d/11bwgPFVkps9AH2NOD1zBDdpF_tQghAB-/view?usp=sharing
You can download the target net (target branch of SiamRPN++) from https://drive.google.com/file/d/1dw_Ne3UMcCnFsaD6xkZepwE4GEpqq7U_/view?usp=sharing
You can download the search net (search branch of SiamRPN++) from https://drive.google.com/file/d/1Lt4oE43ZSucJvze3Y-Z87CVDreO-Afwl/view?usp=sharing
You can download the head model (RPN Head) from https://drive.google.com/file/d/1zT1yu12mtj3JQEkkfKFJWiZ71fJ-dQTi/view?usp=sharing
"""

class ModelBuilder():
    """ This class generates the SiamRPN++ Tracker Model by using Imported ONNX Nets
    """
    def __init__(self, target_net, search_net, rpn_head):
        super(ModelBuilder, self).__init__()
        # Build the target branch
        self.target_net = target_net
        # Build the search branch
        self.search_net = search_net
        # Build RPN_Head
        self.rpn_head = rpn_head

    def template(self, z):
        """ Takes the template of size (1, 1, 127, 127) as an input to generate kernel
        """
        self.target_net.setInput(z)
        outNames = self.target_net.getUnconnectedOutLayersNames()
        self.zfs_1, self.zfs_2, self.zfs_3 = self.target_net.forward(outNames)

    def track(self, x):
        """ Takes the search of size (1, 1, 255, 255) as an input to generate classification score and bounding box regression
        """
        self.search_net.setInput(x)
        outNames = self.search_net.getUnconnectedOutLayersNames()
        xfs_1, xfs_2, xfs_3 = self.search_net.forward(outNames)
        self.rpn_head.setInput(np.stack([self.zfs_1, self.zfs_2, self.zfs_3]), 'input_1')
        self.rpn_head.setInput(np.stack([xfs_1, xfs_2, xfs_3]), 'input_2')
        outNames = self.rpn_head.getUnconnectedOutLayersNames()
        cls, loc = self.rpn_head.forward(outNames)
        return {'cls': cls, 'loc': loc}

class Anchors:
    """ This class generate anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size
        self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchors = self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride**2
        count = 0
        for r in self.ratios:
            ws = int(np.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1
        return anchors

class SiamRPNppTracker:
    def __init__(self, is_cuda, target_net_file = None, search_net_file = None, rpn_head_file = None):
        super(SiamRPNppTracker, self).__init__()
        self.anchor_stride = 8
        self.anchor_ratios = [0.33, 0.5, 1, 2, 3]
        self.anchor_scales = [8]
        self.track_base_size = 8
        self.track_context_amount = 0.5
        self.track_exemplar_size = 127
        self.track_instance_size = 255
        self.track_lr = 0.4
        self.track_penalty_k = 0.04
        self.track_window_influence = 0.44
        self.score_size = (self.track_instance_size - self.track_exemplar_size) // \
                          self.anchor_stride + 1 + self.track_base_size
        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)

        if target_net_file is None:
            target_net_file = os.path.join(os.path.dirname(__file__), "models/target_net.onnx")
        if search_net_file is None:
            search_net_file = os.path.join(os.path.dirname(__file__), "models/search_net.onnx")
        if rpn_head_file is None:
            rpn_head_file = os.path.join(os.path.dirname(__file__), "models/rpn_head.onnx")

        target_net = cv.dnn.readNetFromONNX(target_net_file)
        search_net = cv.dnn.readNetFromONNX(search_net_file)
        rpn_head = cv.dnn.readNetFromONNX(rpn_head_file)

        if is_cuda:
            target_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            target_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
            search_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            search_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
            rpn_head.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            rpn_head.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
        else:
            target_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            target_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            search_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            search_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            rpn_head.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            rpn_head.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.model = ModelBuilder(target_net, search_net, rpn_head)

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        Args:
            im:         bgr based input image frame
            pos:        position of the center of the frame
            model_sz:   exemplar / target image size
            s_z:        original / search image size
            avg_chans:  channel average
        Return:
            im_patch:   sub_windows for the given image input
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_h, im_w, im_d = im.shape
        c = (original_sz + 1) / 2
        cx, cy = pos
        context_xmin = np.floor(cx - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(cy - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_w + 1))
        bottom_pad = int(max(0., context_ymax - im_h + 1))
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im_d)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + im_w, :] = avg_chans
            if bottom_pad:
                te_im[im_h + top_pad:, left_pad:left_pad + im_w, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, im_w + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def generate_anchor(self, score_size):
        """
        Args:
            im:         bgr based input image frame
            pos:        position of the center of the frame
            model_sz:   exemplar / target image size
            s_z:        original / search image size
            avg_chans:  channel average
        Return:
            anchor:     anchors for pre-determined values of stride, ratio, and scale
        """
        anchors = Anchors(self.anchor_stride, self.anchor_ratios, self.anchor_scales)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchors.anchor_num
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        """
        Args:
            delta:      localisation
            anchor:     anchor of pre-determined anchor size
        Return:
            delta:      prediction of bounding box
        """
        delta_transpose = np.transpose(delta, (1, 2, 3, 0))
        delta_contig = np.ascontiguousarray(delta_transpose)
        delta = delta_contig.reshape(4, -1)
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _softmax(self, x):
        """
        Softmax in the direction of the depth of the layer
        """
        x = x.astype(dtype=np.float32)
        x_max = x.max(axis=1)[:, np.newaxis]
        e_x = np.exp(x-x_max)
        div = np.sum(e_x, axis=1)[:, np.newaxis]
        y = e_x / div
        return y

    def _convert_score(self, score):
        """
        Args:
            cls:        score
        Return:
            cls:        score for cls
        """
        score_transpose = np.transpose(score, (1, 2, 3, 0))
        score_con = np.ascontiguousarray(score_transpose)
        score_view = score_con.reshape(2, -1)
        score = np.transpose(score_view, (1, 0))
        score = self._softmax(score)
        return score[:,1]

    def _bbox_clip(self, cx, cy, width, height, boundary):
        """
        Adjusting the bounding box
        """
        bbox_h, bbox_w = boundary
        cx = max(0, min(cx, bbox_w))
        cy = max(0, min(cy, bbox_h))
        width = max(10, min(width, bbox_w))
        height = max(10, min(height, bbox_h))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        Args:
            img(np.ndarray):    bgr based input image frame
            bbox: (x, y, w, h): bounding box
        """
        x, y, w, h = bbox
        self.center_pos = np.array([x + (w - 1) / 2, y + (h - 1) / 2])
        self.h = h
        self.w = w
        w_z = self.w + self.track_context_amount * np.add(h, w)
        h_z = self.h + self.track_context_amount * np.add(h, w)
        s_z = round(np.sqrt(w_z * h_z))
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos, self.track_exemplar_size, s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        Args:
            img(np.ndarray): BGR image
        Return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.w + self.track_context_amount * np.add(self.w, self.h)
        h_z = self.h + self.track_context_amount * np.add(self.w, self.h)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.track_exemplar_size / s_z
        s_x = s_z * (self.track_instance_size / self.track_exemplar_size)
        x_crop = self.get_subwindow(img, self.center_pos, self.track_instance_size, round(s_x), self.channel_average)
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.w * scale_z, self.h * scale_z)))

        # aspect ratio penalty
        r_c = change((self.w / self.h) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.track_penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.track_window_influence) + \
                 self.window * self.track_window_influence
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.track_lr

        cpx, cpy = self.center_pos
        x,y,w,h = bbox
        cx = x + cpx
        cy = y + cpy

        # smooth bbox
        width = self.w * (1 - lr) + w * lr
        height = self.h * (1 - lr) + h * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.w = width
        self.h = height
        bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = score[best_idx]
        return {'bbox': bbox, 'best_score': best_score}
