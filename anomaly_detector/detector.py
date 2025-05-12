from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.models import resnet50, ResNet50_Weights


class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1]).eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.transform = ResNet50_Weights.DEFAULT.transforms()

    def extract(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().cpu()


class MemoryBank:
    def __init__(self, k=50, top_n_features=500):
        self.k = k
        self.top_n = top_n_features
        self.bank = None
        self.selected_indices = None
        self.threshold = None

    def build(self, features):
        std = features.std(dim=0)
        _, indices = torch.sort(std)
        self.selected_indices = indices[-self.top_n:]
        self.bank = features[:, self.selected_indices]

    def compute_threshold(self, features, n_sigma=2):
        distances = []
        for feat in features:
            dist = torch.norm(self.bank - feat[self.selected_indices], dim=1)
            dist = dist.topk(self.k, largest=False).values.mean()
            distances.append(dist.item())
        self.threshold = np.mean(distances) + n_sigma * np.std(distances)
        return self.threshold

    def compute_score(self, feature):
        dist = torch.norm(self.bank - feature[self.selected_indices], dim=1)
        dist = dist.topk(self.k, largest=False).values.mean()
        return dist.item()


class AnomalyDetector:
    def __init__(self, data_root, device='cpu'):
        self.device = device
        self.data_root = Path(data_root)
        self.feature_extractor = FeatureExtractor(device)
        self.memory_bank = MemoryBank()
        self.classes = [p.name for p in (self.data_root / 'test').iterdir() if p.is_dir()]

    def train(self):
        train_dir = self.data_root / 'train' / 'good'
        features = [self.feature_extractor.extract(p) for p in tqdm(train_dir.glob('*.*'), desc="Extracting Train Features")]
        features = torch.stack(features)
        self.memory_bank.build(features)
        self.memory_bank.compute_threshold(features)

    def evaluate(self):
        y_true, y_scores = [], []
        for class_name in self.classes:
            label = 0 if class_name == 'good' else 1
            for p in (self.data_root / 'test' / class_name).glob('*.*'):
                feat = self.feature_extractor.extract(p)
                score = self.memory_bank.compute_score(feat)
                y_true.append(label)
                y_scores.append(score)
        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        return {"roc_auc": auc, "fpr": fpr, "tpr": tpr, "y_true": y_true, "y_score": y_scores}

    def predict(self, image_path):
        feat = self.feature_extractor.extract(image_path)
        score = self.memory_bank.compute_score(feat)
        return {"anomaly": score > self.memory_bank.threshold, "score": score}

    def visualize_tsne(self):
        features, labels = [], []
        for class_name in self.classes:
            for p in (self.data_root / 'test' / class_name).glob('*.*'):
                feat = self.feature_extractor.extract(p)
                features.append(feat.numpy())
                labels.append(class_name)
        features_2d = TSNE(n_components=2, random_state=42).fit_transform(np.array(features))
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels)
        plt.title("t-SNE Feature Visualization")
        return plt
