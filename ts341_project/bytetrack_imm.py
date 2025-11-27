# bytetrack_imm_kalman.py
# Requirements: numpy, opencv-python (only if use_appearance=True)
import numpy as np
import cv2


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def bbox_wh(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([x2 - x1, y2 - y1])


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1); inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    a1 = max(1e-6, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(1e-6, (b2[2]-b2[0])*(b2[3]-b2[1]))
    return inter / (a1 + a2 - inter + 1e-9)


class Kalman:
    """
    Simple Kalman for 2D position+velocity (state = [x, y, vx, vy]).
    Works as a building block for CV model. For CA model we will augment state.
    """
    def __init__(self, init_center, dt=1.0,
                 process_noise_scale=1.0, measure_noise_scale=10.0):
        # state: [x, y, vx, vy]
        self.state = np.array([init_center[0], init_center[1], 0.0, 0.0], dtype=float)

        # Transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)

        # Observation matrix (we measure x,y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Covariances
        self.P = np.eye(4) * 200.0  # initial uncertainty
        self.Q = np.eye(4) * process_noise_scale
        self.R = np.eye(2) * measure_noise_scale

    def predict(self):
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.state.copy()

    def update(self, meas_center):
        z = np.array([meas_center[0], meas_center[1]])
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.state = self.state + K.dot(y)
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def mahalanobis_distance(self, meas_center):
        z = np.array([meas_center[0], meas_center[1]])
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        return float(y.T.dot(np.linalg.inv(S)).dot(y))


class KalmanCA:
    """
    Very simple CA (constant acceleration) Kalman for [x,y,vx,vy,ax,ay].
    Not fully tuned — used in parallel with CV.
    """
    def __init__(self, init_center, dt=1.0,
                 process_noise_scale=1.0, measure_noise_scale=10.0):
        # state: [x, y, vx, vy, ax, ay]
        self.state = np.array([init_center[0], init_center[1], 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.dt = dt
        # Transition F for CA
        dt2 = 0.5 * dt * dt
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        # observe x,y
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,1]=1

        self.P = np.eye(6) * 300.0
        self.Q = np.eye(6) * process_noise_scale
        self.R = np.eye(2) * measure_noise_scale

    def predict(self):
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.state.copy()

    def update(self, meas_center):
        z = np.array([meas_center[0], meas_center[1]])
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.state = self.state + K.dot(y)
        I = np.eye(6)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def mahalanobis_distance(self, meas_center):
        z = np.array([meas_center[0], meas_center[1]])
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        return float(y.T.dot(np.linalg.inv(S)).dot(y))


class Track:
    """
    Track holding both CV and CA Kalman filters and IMM mixing probabilities.
    """
    def __init__(self, track_id, init_bbox, use_appearance=False, frame=None):
        self.track_id = track_id
        self.bbox = list(init_bbox)
        self.center = bbox_center(init_bbox)
        self.size = bbox_wh(init_bbox)

        # Kalman models
        self.kf_cv = Kalman(self.center.copy())
        self.kf_ca = KalmanCA(self.center.copy())

        # model probabilities (uniform init)
        self.prob_cv = 0.5
        self.prob_ca = 0.5

        # lifetime info
        self.age = 0
        self.time_since_update = 0
        self.hits = 1

        # appearance (optional)
        self.use_appearance = use_appearance
        self.appearance_hist = None
        if use_appearance and frame is not None:
            self.update_appearance(frame, init_bbox)

    def predict(self):
        # call predict of both models
        s_cv = self.kf_cv.predict()      # returns state vector
        s_ca = self.kf_ca.predict()
        # combine predicted centers via probabilities
        pred_cv_center = s_cv[0:2]
        pred_ca_center = s_ca[0:2]
        pred_center = self.prob_cv * pred_cv_center + self.prob_ca * pred_ca_center
        # update bbox center without changing size
        w, h = self.size
        self.bbox = [pred_center[0] - w/2, pred_center[1] - h/2,
                     pred_center[0] + w/2, pred_center[1] + h/2]
        self.center = pred_center
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, new_bbox, frame=None):
        # measurement center
        meas_center = bbox_center(new_bbox)
        # update each kalman
        self.kf_cv.update(meas_center)
        self.kf_ca.update(meas_center)
        # update model probabilities according to Mahalanobis distance (lower=better)
        d_cv = self.kf_cv.mahalanobis_distance(meas_center)
        d_ca = self.kf_ca.mahalanobis_distance(meas_center)
        # convert distances to likelihood-like scores
        score_cv = np.exp(-0.5 * d_cv)
        score_ca = np.exp(-0.5 * d_ca)
        s = score_cv + score_ca + 1e-9
        self.prob_cv = score_cv / s
        self.prob_ca = score_ca / s
        # set bbox, center and size
        self.bbox = list(new_bbox)
        self.center = meas_center
        self.size = bbox_wh(new_bbox)
        # appearance
        if self.use_appearance and frame is not None:
            self.update_appearance(frame, new_bbox)
        # bookkeeping
        self.time_since_update = 0
        self.hits += 1

    def update_appearance(self, frame, bbox):
        x1,y1,x2,y2 = [int(max(0, v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        hist = cv2.calcHist([crop], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        self.appearance_hist = hist

    def appearance_similarity(self, hist):
        if self.appearance_hist is None or hist is None:
            return 0.0
        # Bhattacharyya-like (1 - distance)
        return float(1.0 - cv2.compareHist(self.appearance_hist.astype('float32'), hist.astype('float32'), cv2.HISTCMP_BHATTACHARYYA))


class IMMByteTrack:
    """
    IMM tracker that:
     - predicts tracks (CV + CA)
     - matches detections with a hybrid score:
         score = w_iou * IoU  + w_motion * exp(-0.5*mahal) + w_app * appearance_sim
       Higher score = better match.
     - updates matched tracks, creates new tracks for unmatched detections,
       prunes old tracks.
    """
    def __init__(self, max_age=60, iou_weight=0.6, motion_weight=0.3, appearance_weight=0.1,
                 iou_threshold=0.15, use_appearance=False):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_weight = iou_weight
        self.motion_weight = motion_weight
        self.appearance_weight = appearance_weight
        self.iou_threshold = iou_threshold
        self.use_appearance = use_appearance

    def _appearance_hist_from_bbox(self, frame, bbox):
        x1,y1,x2,y2 = [int(max(0, v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hist = cv2.calcHist([crop], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        return cv2.normalize(hist, hist).flatten()

    def _match_score(self, track, det_bbox, det_hist=None):
        # IoU in [0,1]
        iou_val = iou(track.bbox, det_bbox)
        # Mahalanobis from both models: combine via track probs
        meas_center = bbox_center(det_bbox)
        # weighted Mahalanobis (lower is better)
        m_cv = track.kf_cv.mahalanobis_distance(meas_center)
        m_ca = track.kf_ca.mahalanobis_distance(meas_center)
        mahal = track.prob_cv * m_cv + track.prob_ca * m_ca
        motion_score = np.exp(-0.5 * mahal)  # in (0,1]
        app_score = 0.0
        if self.use_appearance and det_hist is not None:
            app_score = track.appearance_similarity(det_hist)
        # final score
        score = self.iou_weight * iou_val + self.motion_weight * motion_score + self.appearance_weight * app_score
        return score

    def predict_all(self):
        for tr in self.tracks:
            tr.predict()

    def update(self, detections, frame=None):
        """
        detections: list of [x1,y1,x2,y2] (floats or ints)
        frame: optional image for appearance histograms (BGR)
        returns list of dicts: {'track_id': id, 'bbox': bbox}
        """
        # 1) predict all tracks
        self.predict_all()

        # 2) compute appearance histograms for detections if needed
        det_hists = None
        if self.use_appearance and frame is not None:
            det_hists = []
            for d in detections:
                det_hists.append(self._appearance_hist_from_bbox(frame, d))
        else:
            det_hists = [None] * len(detections)

        # 3) matching: greedy best-match (fast, simple)
        N_t = len(self.tracks)
        N_d = len(detections)
        matches = []
        unmatched_tracks = set(range(N_t))
        unmatched_detections = set(range(N_d))

        if N_t > 0 and N_d > 0:
            # compute score matrix
            scores = np.zeros((N_t, N_d), dtype=float)
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    scores[i, j] = self._match_score(tr, det, det_hists[j])
            # greedy matching: pick best score > threshold iteratively
            while True:
                idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
                best_score = scores[idx]
                if best_score < self.iou_threshold:
                    break
                t_idx, d_idx = idx
                matches.append((t_idx, d_idx))
                unmatched_tracks.discard(t_idx)
                unmatched_detections.discard(d_idx)
                # invalidate row and col
                scores[t_idx, :] = -1.0
                scores[:, d_idx] = -1.0

        # 4) update matched tracks
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_idx].update(det, frame if self.use_appearance else None)

        # 5) create new tracks for unmatched detections (but require 1 hit to confirm)
        # we'll create new tracks but they can be pruned if not re-detected
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            tr = Track(self.next_id, det, use_appearance=self.use_appearance, frame=frame)
            self.next_id += 1
            self.tracks.append(tr)

        # 6) increase age/time_since_update for unmatched tracks and prune old ones
        survivors = []
        for tr in self.tracks:
            if tr.time_since_update < self.max_age:
                survivors.append(tr)
        self.tracks = survivors

        # 7) return active track list
        return [{'track_id': tr.track_id, 'bbox': tr.bbox} for tr in self.tracks]
