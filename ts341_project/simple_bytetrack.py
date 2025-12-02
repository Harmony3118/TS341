import numpy as np

class Track:
    """Représente un objet suivi avec un IMM multi-modèles."""
    def __init__(self, track_id, init_bbox):
        self.track_id = track_id
        self.bbox = init_bbox  # [x1, y1, x2, y2]
        cx = (init_bbox[0] + init_bbox[2]) / 2
        cy = (init_bbox[1] + init_bbox[3]) / 2
        # état par modèle
        self.state_cv = np.array([cx, cy, 0.0, 0.0])        # [pos_x, pos_y, vel_x, vel_y]
        self.state_ca = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0])  # [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        # poids des modèles
        self.prob_cv = 0.5
        self.prob_ca = 0.5
        self.age = 0   # frames depuis la dernière détection
        self.time_since_update = 0

    def predict(self):
        """Prédit la position du track avec l’IMM pondéré."""
        # CV
        pos_cv = self.state_cv[:2] + self.state_cv[2:4]
        # CA
        pos_ca = self.state_ca[:2] + self.state_ca[2:4] + 0.5*self.state_ca[4:6]
        # Position finale pondérée
        pred_pos = self.prob_cv * pos_cv + self.prob_ca * pos_ca
        return pred_pos

    def update(self, new_bbox):
        """Met à jour les états et les probabilités des modèles avec une nouvelle détection."""
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2

        # Update CV
        vel_x = new_cx - self.state_cv[0]
        vel_y = new_cy - self.state_cv[1]
        self.state_cv = np.array([new_cx, new_cy, vel_x, vel_y])

        # Update CA
        acc_x = (new_cx - self.state_ca[0] - self.state_ca[2])
        acc_y = (new_cy - self.state_ca[1] - self.state_ca[3])
        vel_x = new_cx - self.state_ca[0]
        vel_y = new_cy - self.state_ca[1]
        self.state_ca = np.array([new_cx, new_cy, vel_x, vel_y, acc_x, acc_y])

        # Mise à jour simple des probabilités selon modèle le plus proche
        err_cv = np.linalg.norm(np.array([new_cx, new_cy]) - (self.state_cv[:2] + self.state_cv[2:4]))
        err_ca = np.linalg.norm(np.array([new_cx, new_cy]) - (self.state_ca[:2] + self.state_ca[2:4] + 0.5*self.state_ca[4:6]))
        total = err_cv + err_ca + 1e-6
        self.prob_cv = (1 - err_cv/total)
        self.prob_ca = (1 - err_ca/total)
        # Normaliser
        s = self.prob_cv + self.prob_ca
        self.prob_cv /= s
        self.prob_ca /= s

        self.bbox = new_bbox
        self.age = 0
        self.time_since_update = 0

class SimpleTrack:
    """Tracker multi-objets avec IMM CV + CA."""
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def _iou(self, bbox1, bbox2):
        """Intersection over union entre deux boîtes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_area = max(0, x2-x1) * max(0, y2-y1)
        area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
        return inter_area / (area1 + area2 - inter_area + 1e-6)

    def update(self, detections):
        """
        detections : liste de bbox [[x1,y1,x2,y2], ...]
        Retourne une liste de tracks avec ID et bbox mises à jour.
        """

        # Si aucun track existant, créer de nouveaux tracks
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1
            return [{'track_id': tr.track_id, 'bbox': tr.bbox} for tr in self.tracks]

        # Calculer IoU entre tracks prédits et detections
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        matches = []

        for t_idx, track in enumerate(self.tracks):
            best_iou = 0
            best_d_idx = -1
            for d_idx, det in enumerate(detections):
                if d_idx not in unmatched_detections:
                    continue
                iou_val = self._iou(track.bbox, det)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_d_idx = d_idx
            if best_iou > self.iou_threshold:
                matches.append((t_idx, best_d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(best_d_idx)

        # Update matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            self.tracks.append(Track(self.next_id, detections[d_idx]))
            self.next_id += 1

        # Increment age for unmatched tracks and remove old tracks
        to_delete = []
        for i, tr in enumerate(self.tracks):
            tr.age += 1
            tr.time_since_update += 1
            if tr.age > self.max_age:
                to_delete.append(i)
        for idx in reversed(to_delete):
            del self.tracks[idx]

        return [{'track_id': tr.track_id, 'bbox': tr.bbox} for tr in self.tracks]
