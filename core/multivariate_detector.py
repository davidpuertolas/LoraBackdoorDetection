import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import safetensors.torch as st


class MultivariateBackdoorDetector:
    """
    Detector that uses a multivariate feature vector (20-dim) from all four projections.
    Trains a logistic regression classifier with regularization and feature scaling.
    """

    def __init__(self, bank=None, model_path: Optional[str] = None):
        self.bank = bank  # unused
        self.classifier = None
        self.scaler = None
        self.threshold = 0.5

        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Return the learned coefficients (20-dim vector) as a flat array."""
        if self.classifier is None:
            return None
        return self.classifier.coef_.flatten()

    @property
    def intercept(self) -> Optional[float]:
        """Return the intercept of the logistic regression."""
        if self.classifier is None:
            return None
        return self.classifier.intercept_[0]

    @staticmethod
    def _select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, str]:
        """
        Select a threshold using the paper's rule:
        - if validation scores are perfectly separated, place the threshold
          close to the benign boundary to preserve recall under mild shift;
        - otherwise fall back to Youden's J statistic.
        """
        benign_scores = y_score[y_true == 0]
        poison_scores = y_score[y_true == 1]

        if len(benign_scores) > 0 and len(poison_scores) > 0:
            benign_max = float(np.max(benign_scores))
            poison_min = float(np.min(poison_scores))
            if benign_max < poison_min:
                separation = poison_min - benign_max
                return benign_max + 0.25 * separation, "perfect_separation_margin"

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden = tpr - fpr
        best_idx = int(np.argmax(youden))
        return float(thresholds[best_idx]), "youden_j"

    def calibrate(self, poison_paths: List[str], benign_paths: List[str],
                  layer_idx: int = 20, val_split: float = 0.2,
                  C: float = 0.1, random_state: int = 42,
                  train_on_val: bool = False) -> Dict[str, Any]:
        """
        Train logistic regression with regularization and find optimal threshold.

        Args:
            poison_paths: list of paths to poisoned adapters.
            benign_paths: list of paths to benign adapters.
            layer_idx: layer index to extract features from.
            val_split: fraction of data to use for threshold calibration.
            C: inverse regularization strength (smaller = stronger regularization).
            random_state: seed for reproducibility.
            train_on_val: if True, train on the validation split and
                calibrate threshold on the train split.

        Returns:
            dict with keys: 'new_weights', 'new_threshold', 'auc',
            'benign_scores', 'poison_scores', and train/val splits.
        """
        print("Extracting features from benign adapters...")
        benign_features = []
        benign_valid_paths = []
        for path in benign_paths:
            feat = self._extract_features_from_adapter(Path(path), layer_idx)
            if feat is not None:
                benign_features.append(feat)
                benign_valid_paths.append(path)
        print(f"Extracted {len(benign_features)} benign feature vectors.")

        print("Extracting features from poisoned adapters...")
        poison_features = []
        poison_valid_paths = []
        for path in poison_paths:
            feat = self._extract_features_from_adapter(Path(path), layer_idx)
            if feat is not None:
                poison_features.append(feat)
                poison_valid_paths.append(path)
        print(f"Extracted {len(poison_features)} poisoned feature vectors.")

        if len(benign_features) == 0 or len(poison_features) == 0:
            raise ValueError("No valid features extracted.")

        # Create dataset
        X = np.vstack(benign_features + poison_features)
        y = np.hstack([np.zeros(len(benign_features)), np.ones(len(poison_features))])
        paths = benign_valid_paths + poison_valid_paths

        # Shuffle and split into train/val
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        split = int(len(X) * (1 - val_split))
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Optionally swap roles for training vs threshold calibration
        if train_on_val:
            X_train, X_val = X_val, X_train
            y_train, y_val = y_val, y_train

        # Standardize features (fit on train only!)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train logistic regression with regularization
        clf = LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=random_state)
        clf.fit(X_train_scaled, y_train)

        # Predict probabilities on train/validation sets
        y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = clf.predict_proba(X_val_scaled)[:, 1]

        # Match the paper: use a margin from the benign boundary whenever
        # validation scores are perfectly separated.
        best_threshold, threshold_mode = self._select_threshold(y_val, y_val_proba)

        # Compute AUC on full dataset (scaled with the same scaler)
        X_scaled = scaler.transform(X)
        y_proba = clf.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_proba)

        # Store classifier, scaler, threshold
        self.classifier = clf
        self.scaler = scaler
        self.threshold = best_threshold

        # Scores for plotting (full, train, val)
        benign_scores = y_proba[y == 0].tolist()
        poison_scores = y_proba[y == 1].tolist()

        benign_scores_train = y_train_proba[y_train == 0].tolist()
        poison_scores_train = y_train_proba[y_train == 1].tolist()
        benign_scores_val = y_val_proba[y_val == 0].tolist()
        poison_scores_val = y_val_proba[y_val == 1].tolist()

        split_manifest = {
            'benign': {
                'train': [paths[i] for i in train_idx if y[i] == 0],
                'val': [paths[i] for i in val_idx if y[i] == 0],
                'test': []
            },
            'poison': {
                'train': [paths[i] for i in train_idx if y[i] == 1],
                'val': [paths[i] for i in val_idx if y[i] == 1],
                'test': []
            }
        }

        return {
            'new_weights': clf.coef_.flatten().tolist(),
            'new_threshold': best_threshold,
            'auc': auc,
            'benign_scores': benign_scores,
            'poison_scores': poison_scores,
            'benign_scores_train': benign_scores_train,
            'poison_scores_train': poison_scores_train,
            'benign_scores_val': benign_scores_val,
            'poison_scores_val': poison_scores_val,
            'intercept': clf.intercept_.tolist()[0],
            'threshold_mode': threshold_mode,
            'split_manifest': split_manifest
        }

    def scan(self, adapter_path: str, layer_idx: int = 20) -> Dict[str, Any]:
        """
        Scan a single adapter and return detection result.

        Args:
            adapter_path: path to adapter directory.
            layer_idx: layer index to extract features.

        Returns:
            dict with keys: 'score' (probability of poison), 'prediction', 'threshold', etc.
        """
        if self.classifier is None or self.scaler is None:
            raise RuntimeError("Detector not calibrated. Run calibrate() first.")

        feat = self._extract_features_from_adapter(Path(adapter_path), layer_idx)
        if feat is None:
            return {'error': 'Feature extraction failed', 'score': None, 'prediction': None}

        # Scale features using fitted scaler
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))

        # Predict probability
        proba = self.classifier.predict_proba(feat_scaled)[0, 1]
        pred = int(proba >= self.threshold)

        return {
            'score': proba,
            'prediction': pred,
            'threshold': self.threshold,
            'features': feat.tolist()
        }

    def save(self, path: str):
        """Save classifier, scaler and threshold to file."""
        data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'threshold': self.threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load classifier, scaler and threshold from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.threshold = data['threshold']

    @staticmethod
    def _extract_features_from_adapter(adapter_path: Path, layer_idx: int) -> Optional[np.ndarray]:
        """
        Extract feature vector (20-dim) from an adapter.
        Returns None if any required projection is missing.
        """
        safetensors_file = adapter_path / "adapter_model.safetensors"
        if not safetensors_file.exists():
            return None

        try:
            weights = st.load_file(str(safetensors_file))
        except Exception:
            return None

        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        features = []

        for proj in proj_names:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"

            if a_key not in weights or b_key not in weights:
                return None

            A = weights[a_key]
            B = weights[b_key]

            # Ensure correct multiplication order (B @ A)
            if B.shape[1] != A.shape[0]:
                # Try transposing B
                if B.shape[0] == A.shape[0]:
                    B = B.T
                else:
                    return None

            metrics = MultivariateBackdoorDetector._compute_metrics_from_matrices(B, A)
            feature_vector = [
                metrics['sigma1'],
                metrics['frobenius_norm'],
                metrics['energy_concentration'],
                metrics['entropy'],
                metrics['kurtosis']
            ]
            features.extend(feature_vector)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _compute_metrics_from_matrices(B: torch.Tensor, A: torch.Tensor) -> Dict[str, float]:
        """
        Compute the five spectral metrics from LoRA matrices B and A.
        Uses full matrix for kurtosis. Efficient SVD via QR.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = B.to(device)
        A = A.to(device)

        # Efficient SVD for low-rank product B@A using QR
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        M = Rb @ Ra.T  # r×r
        s = torch.linalg.svdvals(M)

        sigma1 = s[0].item()
        frob_norm = torch.sqrt(torch.sum(s**2)).item()
        total_energy = torch.sum(s**2).item()
        energy_conc = (s[0].item()**2) / total_energy if total_energy > 0 else 0.0

        p = s / (torch.sum(s) + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()

        # Compute full ΔW for kurtosis
        delta = B @ A  # d×d

        # Flatten and compute kurtosis
        flat = delta.flatten().to(torch.float64)
        mean = torch.mean(flat)
        var = torch.var(flat)
        if var > 0:
            kurt = (torch.mean((flat - mean)**4) / (var**2)).item() - 3.0
        else:
            kurt = 0.0

        # Clean up
        del B, A, Qb, Rb, Qa, Ra, M, s, delta, flat
        torch.cuda.empty_cache()

        return {
            'sigma1': sigma1,
            'frobenius_norm': frob_norm,
            'energy_concentration': energy_conc,
            'entropy': entropy,
            'kurtosis': kurt
        }
