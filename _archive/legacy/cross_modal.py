"""
CrossModalCodes — интерфейс для будущих модальностей.

Позволяет кодировать изображения, аудио и другие модальности
в то же латентное пространство R^k, что и текстовые коды.
После этого все механизмы (State Algebra, KCA, SRG) работают без изменений.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from loguru import logger


class CrossModalEncoder(nn.Module):
    """
    Базовый энкодер для произвольной модальности.

    Наследуйте для конкретных типов данных (изображения, аудио).
    """

    def __init__(self, input_dim: int, latent_dim: int = 2560):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8)


class ImageEncoder(CrossModalEncoder):
    """
    Энкодер изображений в латентное пространство.
    Вход: плоский вектор пикселей или эмбеддинги от предобученной CNN.
    """

    def __init__(self, input_dim: int = 2048, latent_dim: int = 2560):
        super().__init__(input_dim, latent_dim)


class AudioEncoder(CrossModalEncoder):
    """
    Энкодер аудио в латентное пространство.
    Вход: спектрограмма или эмбеддинги от предобученной модели.
    """

    def __init__(self, input_dim: int = 1024, latent_dim: int = 2560):
        super().__init__(input_dim, latent_dim)


class CrossModalBridge:
    """
    Мост между модальностями: конвертирует входные данные
    в латентный код, совместимый с FCF.

    Использование:
        bridge = CrossModalBridge()
        z_image = bridge.encode_image(pixels)
        z_audio = bridge.encode_audio(spectrogram)

        # Теперь оба кода в одном пространстве — можно применять State Algebra
        z_combined = state_algebra.cross_attend(z_image, z_text)
    """

    def __init__(self, latent_dim: int = 2560):
        self.latent_dim = latent_dim
        self.image_encoder = ImageEncoder(latent_dim=latent_dim)
        self.audio_encoder = AudioEncoder(latent_dim=latent_dim)

    def encode_image(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            z = self.image_encoder(x)
        return z.squeeze(0).numpy()

    def encode_audio(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            z = self.audio_encoder(x)
        return z.squeeze(0).numpy()

    def encode(self, features: np.ndarray, modality: str = "generic") -> np.ndarray:
        if modality == "image":
            return self.encode_image(features)
        elif modality == "audio":
            return self.encode_audio(features)
        else:
            x = torch.from_numpy(features).float().unsqueeze(0)
            with torch.no_grad():
                encoder = CrossModalEncoder(features.shape[-1], self.latent_dim)
                z = encoder(x)
            return z.squeeze(0).numpy()
