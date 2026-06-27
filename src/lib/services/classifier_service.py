from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado.
        Modelo B (opcional, recomendado): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from torchvision import models, transforms
        from PIL import Image
        import numpy as np

        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Dataset
        class DogDataset(Dataset):
            def __init__(self, root: Path, transform=None):
                self.transform = transform
                self.samples: list[tuple[Path, int]] = []
                breeds = sorted([d.name for d in root.iterdir() if d.is_dir()])
                self.class_to_idx = {b: i for i, b in enumerate(breeds)}
                for breed in breeds:
                    for img_path in sorted((root / breed).iterdir()):
                        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            self.samples.append((img_path, self.class_to_idx[breed]))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label

        train_ds = DogDataset(self.dataset_path / "train", train_transform)
        val_ds   = DogDataset(self.dataset_path / "valid", val_transform)
        num_classes = len(train_ds.class_to_idx)

        # Sampler balanceado
        labels      = [label for _, label in train_ds.samples]
        class_counts = np.bincount(labels)
        sample_weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=0)

        # Modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Entrenando {self.active_model_name} en {device}")

        if self.active_model_name == "resnet18_finetuned":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif self.active_model_name == "cnn_custom":
            def conv_block(in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            model = nn.Sequential(
                conv_block(3,   32),
                conv_block(32,  64),
                conv_block(64, 128),
                conv_block(128, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Modelo no soportado: {self.active_model_name}")

        model = model.to(device)


        # Loss con pesos por clase
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Loop entrenamiento
        EPOCHS = 20
        best_val_acc = 0.0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for imgs, labels_batch in train_loader:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                correct      += (out.detach().argmax(1) == labels_batch).sum().item()
                total        += imgs.size(0)
            scheduler.step()

            train_loss = running_loss / total
            train_acc  = correct / total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels_batch in val_loader:
                    imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                    out  = model(imgs)
                    loss = criterion(out, labels_batch)
                    val_loss    += loss.item() * imgs.size(0)
                    val_correct += (out.argmax(1) == labels_batch).sum().item()
                    val_total   += imgs.size(0)

            val_loss = val_loss / val_total
            val_acc  = val_correct / val_total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} — "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model":         model,
                    "class_to_idx":  train_ds.class_to_idx,
                    "history":       history,
                    "image_size":    self.image_size,
                    "model_name":    self.active_model_name,
                }, self.active_checkpoint)
                logger.info(f"  ✓ Checkpoint guardado (val_acc={val_acc:.4f})")

        logger.info(f"Entrenamiento finalizado. Mejor val_acc: {best_val_acc:.4f}")

  

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en la notebook.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
        import numpy as np

        checkpoint   = torch.load(self.active_checkpoint, map_location="cpu", weights_only=False)
        model        = checkpoint["model"]
        class_to_idx = checkpoint["class_to_idx"]
        image_size   = checkpoint.get("image_size", self.image_size)

        # Guardamos en self para que el notebook los grafique sin recargar
        self.history          = checkpoint.get("history", {})
        self.class_to_idx_    = class_to_idx
        self.idx_to_class_    = {v: k for k, v in class_to_idx.items()}
        num_classes           = len(class_to_idx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)
        model.eval()

        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class DogDataset(Dataset):
            def __init__(self, root: Path, class_to_idx: dict, transform=None):
                self.transform = transform
                self.samples: list[tuple[Path, int]] = []
                for breed, idx in class_to_idx.items():
                    breed_dir = root / breed
                    if not breed_dir.exists():
                        continue
                    for img_path in sorted(breed_dir.iterdir()):
                        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            self.samples.append((img_path, idx))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label

        test_ds     = DogDataset(self.dataset_path / "test", class_to_idx, val_transform)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(device)
                preds  = model(imgs).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy  = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        specificities = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificities))

        
        self.confusion_matrix_ = cm
        self.class_names_      = [self.idx_to_class_[i] for i in range(num_classes)]

        metrics = {
            "accuracy":    round(float(accuracy),    4),
            "precision":   round(float(precision),   4),
            "recall":      round(float(recall),       4),
            "specificity": round(float(specificity), 4),
            "f1":          round(float(f1),          4),
        }
        logger.info(f"evaluate_classifier [{self.active_model_name}]: {metrics}")

        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from PIL import Image
        import cv2

        checkpoint   = self.load_model()
        model        = checkpoint["model"]
        image_size   = checkpoint.get("image_size", self.image_size)

        # Extraer penúltima capa según arquitectura
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.active_model_name == "resnet18_finetuned":
            # Quitamos la fc final → embedding de 512 dims
            embedding_model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        elif self.active_model_name == "cnn_custom":
            # Quitamos la Linear final (último elemento del Sequential)
            embedding_model = model[:-1]   # hasta Dropout inclusive → 256 dims
        else:
            raise ValueError(f"Modelo no soportado: {self.active_model_name}")

        embedding_model = embedding_model.to(device)
        embedding_model.eval()

        # Imagen BGR (OpenCV) → RGB → tensor normalizado
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img    = Image.fromarray(img_rgb)
        tensor     = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = embedding_model(tensor).squeeze().cpu().numpy()

        return embedding.tolist()
