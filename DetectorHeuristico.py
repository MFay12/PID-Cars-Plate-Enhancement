"""Detector heurístico de placas baseado em visão computacional clássica.

Os candidatos são expostos no formato usado por este repositório:
``(score, x, y, largura, altura)``.
"""

import math

import cv2
import numpy as np


FAMILIAS_PLACA = {
    "carro": {
        "name": "horizontal",
        "ratio_min": 2.35,
        "ratio_max": 3.65,
        "ratio_target": 2.90,
        "ratio_sigma": 0.46,
        "area_min": 0.005,
        "area_max": 0.11,
        "area_target": 0.022,
        "kernel_variants_640": (
            (13, 5, None, None),
            (21, 5, None, None),
        ),
        "expansions": ((0.06, 0.18), (0.12, 0.32)),
    },
    "moto": {
        "name": "motorcycle",
        "ratio_min": 0.85,
        "ratio_max": 1.60,
        "ratio_target": 1.15,
        "ratio_sigma": 0.26,
        "area_min": 0.008,
        "area_max": 0.16,
        "area_target": 0.045,
        "kernel_variants_640": (
            (13, 5, (31, 35), None),
            (13, 5, None, ((0.12, 0.25), (0.20, 0.45))),
        ),
        "expansions": ((0.06, 0.03), (0.12, 0.08)),
    },
}


class DetectorHeuristicoPlaca:
    """Localiza e ordena placas usando visão clássica e score composto."""

    def __init__(
        self,
        score_threshold=0.5,
        normalized_width=512,
        clahe_clip_limit=3.0,
        min_area_ratio=0.003,
        max_area_ratio=0.16,
        max_candidates=24,
        adaptive_threshold_fallback=True,
    ):
        self.score_threshold = float(score_threshold)
        self.normalized_width = max(320, int(normalized_width))
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.max_candidates = max(4, int(max_candidates))
        self.adaptive_threshold_fallback = bool(adaptive_threshold_fallback)
        self.last_debug = {}

    @staticmethod
    def _odd(value, minimum=3):
        value = max(minimum, int(value))
        return value if value % 2 else value + 1

    @staticmethod
    def _gaussian_score(value, target, sigma):
        if sigma <= 0:
            return 0.0
        return math.exp(-0.5 * ((value - target) / sigma) ** 2)

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / float(union) if union else 0.0

    def _scaled_kernel(self, width, height):
        scale = self.normalized_width / 640.0
        return cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                self._odd(round(width * scale), 3),
                self._odd(round(height * scale), 3),
            ),
        )

    def _directional_character_mask(self, enhanced, light_regions, kernel):
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        gradient_x = np.absolute(cv2.Scharr(blackhat, cv2.CV_32F, 1, 0))
        minimum = float(gradient_x.min())
        maximum = float(gradient_x.max())
        gradient_x = (
            (gradient_x - minimum) / (maximum - minimum + 1e-6) * 255.0
        ).astype("uint8")
        gradient_x = cv2.GaussianBlur(gradient_x, (5, 5), 0)
        gradient_x = cv2.morphologyEx(
            gradient_x, cv2.MORPH_CLOSE, kernel
        )
        _, mask = cv2.threshold(
            gradient_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_and(mask, light_regions)

    def _character_features(self, roi, family_name):
        height, width = roi.shape
        roi_area = float(width * height)

        def analyze_mask(mask):
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            components = []
            for index in range(1, component_count):
                x, y, component_width, component_height, area = stats[index]
                if component_height <= 0 or component_width <= 0:
                    continue
                component_ratio = component_width / float(component_height)
                fill_ratio = area / float(component_width * component_height)
                if not (
                    0.28 <= component_height / float(height) <= 0.95
                    and 0.015 <= component_width / float(width) <= 0.38
                    and 0.08 <= component_ratio <= 1.05
                    and 0.12 <= fill_ratio <= 0.92
                    and y > 0
                    and y + component_height < height
                ):
                    continue
                components.append(
                    (x, y, component_width, component_height, area)
                )

            count = len(components)
            consistency = 0.0
            coverage = 0.0
            if components:
                heights = np.asarray(
                    [component[3] for component in components],
                    dtype=np.float32,
                )
                center_ys = np.asarray(
                    [
                        component[1] + component[3] / 2.0
                        for component in components
                    ],
                    dtype=np.float32,
                )
                consistency = self._clip01(
                    1.0
                    - float(np.std(heights))
                    / (float(np.mean(heights)) + 1e-6)
                )
                if family_name == "horizontal":
                    consistency *= self._clip01(
                        1.0
                        - float(np.std(center_ys))
                        / (height * 0.35 + 1e-6)
                    )
                coverage = sum(component[4] for component in components) / roi_area

            score = (
                min(count / 5.0, 1.0)
                * math.exp(-max(0, count - 10) / 5.0)
                * (0.5 + 0.5 * consistency)
            )
            return {
                "count": count,
                "consistency": consistency,
                "coverage": coverage,
                "score": score,
            }

        _, otsu = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        best = analyze_mask(otsu)

        if self.adaptive_threshold_fallback and best["score"] < 0.45:
            block_size = min(
                31, max(3, (min(height, width) // 2) * 2 - 1)
            )
            if block_size >= 3:
                adaptive = cv2.adaptiveThreshold(
                    roi,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    block_size,
                    5,
                )
                alternative = analyze_mask(adaptive)
                if alternative["score"] > best["score"]:
                    best = alternative
        return float(best["score"]), best

    def _candidate_score(self, enhanced, box, family, contour_fill):
        x, y, width, height = box
        image_height, image_width = enhanced.shape
        area_ratio = (width * height) / float(image_width * image_height)
        ratio = width / float(height)

        intensity_roi = enhanced[y : y + height, x : x + width]
        contrast = float(np.std(intensity_roi)) if intensity_roi.size else 0.0
        contrast_score = self._clip01(contrast / 52.0)
        character_score, character_features = self._character_features(
            intensity_roi, family["name"]
        )
        contour_score = self._clip01((contour_fill - 0.20) / 0.60)
        ratio_score = self._gaussian_score(
            ratio, family["ratio_target"], family["ratio_sigma"]
        )
        log_area_error = abs(
            math.log(max(area_ratio, 1e-8) / family["area_target"])
        )
        area_score = self._gaussian_score(log_area_error, 0.0, 1.15)

        center_x = (x + width / 2.0) / float(image_width)
        center_y = (y + height / 2.0) / float(image_height)
        position_score = 0.5 * self._gaussian_score(
            center_x, 0.45, 0.30
        ) + 0.5 * self._gaussian_score(center_y, 0.72, 0.22)

        score = (
            0.18 * ratio_score
            + 0.24 * area_score
            + 0.14 * position_score
            + 0.34 * character_score
            + 0.05 * contrast_score
            + 0.05 * contour_score
        )
        features = {
            "family": family["name"],
            "ratio": ratio,
            "area_ratio": area_ratio,
            "contrast": contrast,
            "contour_fill": contour_fill,
            "character_count": character_features["count"],
            "character_consistency": character_features["consistency"],
            "character_coverage": character_features["coverage"],
            "character_score": character_score,
            "position_x": center_x,
            "position_y": center_y,
            "score": score,
        }
        return self._clip01(score), features

    @staticmethod
    def _prepare_gray(image):
        if image is None or image.size == 0:
            return None
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def buscar(self, image, familias=("carro", "moto"), limite=5):
        """Retorna candidatos ``(score, x, y, largura, altura)`` ordenados."""
        gray = self._prepare_gray(image)
        if gray is None:
            self.last_debug = {}
            return []

        selected_families = []
        for family_key in familias:
            if family_key not in FAMILIAS_PLACA:
                raise ValueError(f"Família de placa desconhecida: {family_key!r}")
            selected_families.append(FAMILIAS_PLACA[family_key])

        original_height, original_width = gray.shape
        scale = self.normalized_width / float(original_width)
        normalized_height = max(1, int(round(original_height * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        normalized = cv2.resize(
            gray,
            (self.normalized_width, normalized_height),
            interpolation=interpolation,
        )

        denoised = cv2.medianBlur(normalized, 3)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(denoised)
        light_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                self._odd(round(self.normalized_width * 9 / 640.0), 3),
                self._odd(round(self.normalized_width * 9 / 640.0), 3),
            ),
        )
        light_regions = cv2.morphologyEx(
            enhanced, cv2.MORPH_CLOSE, light_kernel
        )
        _, light_regions = cv2.threshold(
            light_regions, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        image_area = float(self.normalized_width * normalized_height)
        raw_candidates = []
        raw_boxes = []
        contour_count = 0
        directional_masks = {}

        for family in selected_families:
            for (
                kernel_width,
                kernel_height,
                group_kernel_640,
                expansion_override,
            ) in family["kernel_variants_640"]:
                mask_key = (kernel_width, kernel_height)
                mask = directional_masks.get(mask_key)
                if mask is None:
                    kernel = self._scaled_kernel(kernel_width, kernel_height)
                    mask = self._directional_character_mask(
                        enhanced, light_regions, kernel
                    )
                    directional_masks[mask_key] = mask
                if group_kernel_640:
                    mask = cv2.morphologyEx(
                        mask,
                        cv2.MORPH_CLOSE,
                        self._scaled_kernel(*group_kernel_640),
                    )
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contour_count += len(contours)

                for contour in contours:
                    x, y, width, height = cv2.boundingRect(contour)
                    contour_fill = (
                        float(cv2.contourArea(contour)) / float(width * height)
                        if width > 0 and height > 0
                        else 0.0
                    )
                    expansions = expansion_override or family["expansions"]
                    for padding_x, padding_y in expansions:
                        left = max(0, x - round(width * padding_x))
                        top = max(0, y - round(height * padding_y))
                        right = min(
                            self.normalized_width,
                            x + width + round(width * padding_x),
                        )
                        bottom = min(
                            normalized_height,
                            y + height + round(height * padding_y),
                        )
                        candidate_width = right - left
                        candidate_height = bottom - top
                        if candidate_width < 24 or candidate_height < 8:
                            continue
                        area_ratio = (
                            candidate_width * candidate_height / image_area
                        )
                        ratio = candidate_width / float(candidate_height)
                        if len(raw_boxes) < 100:
                            raw_boxes.append(
                                {
                                    "family": family["name"],
                                    "box": (left, top, right, bottom),
                                    "ratio": ratio,
                                    "area_ratio": area_ratio,
                                }
                            )
                        minimum_area = max(
                            self.min_area_ratio, family["area_min"]
                        )
                        maximum_area = min(
                            self.max_area_ratio, family["area_max"]
                        )
                        if not (
                            minimum_area <= area_ratio <= maximum_area
                            and family["ratio_min"]
                            <= ratio
                            <= family["ratio_max"]
                        ):
                            continue
                        preliminary_score = (
                            0.40
                            * self._gaussian_score(
                                ratio,
                                family["ratio_target"],
                                family["ratio_sigma"],
                            )
                            + 0.45
                            * self._gaussian_score(
                                abs(
                                    math.log(
                                        max(area_ratio, 1e-8)
                                        / family["area_target"]
                                    )
                                ),
                                0.0,
                                1.15,
                            )
                            + 0.15 * self._clip01(contour_fill)
                        )
                        raw_candidates.append(
                            {
                                "box": (left, top, right, bottom),
                                "family": family,
                                "contour_fill": contour_fill,
                                "preliminary_score": preliminary_score,
                            }
                        )

        raw_candidates.sort(
            key=lambda item: item["preliminary_score"], reverse=True
        )
        unique_candidates = []
        for candidate in raw_candidates:
            duplicate = any(
                candidate["family"]["name"] == existing["family"]["name"]
                and self._box_iou(candidate["box"], existing["box"]) >= 0.90
                for existing in unique_candidates
            )
            if not duplicate:
                unique_candidates.append(candidate)
            if len(unique_candidates) >= self.max_candidates:
                break

        candidates = []
        for candidate in unique_candidates:
            left, top, right, bottom = candidate["box"]
            score, features = self._candidate_score(
                enhanced,
                (left, top, right - left, bottom - top),
                candidate["family"],
                candidate["contour_fill"],
            )
            mapped_box = (
                max(0, min(original_width, int(round(left / scale)))),
                max(0, min(original_height, int(round(top / scale)))),
                max(0, min(original_width, int(round(right / scale)))),
                max(0, min(original_height, int(round(bottom / scale)))),
            )
            if mapped_box[2] <= mapped_box[0] or mapped_box[3] <= mapped_box[1]:
                continue
            candidates.append(
                {
                    "box": mapped_box,
                    "score": score,
                    "features": features,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        accepted = [
            candidate
            for candidate in candidates
            if candidate["score"] >= self.score_threshold
        ]
        self.last_debug = {
            "contours": contour_count,
            "geometry_candidates": len(raw_candidates),
            "unique_candidates": len(unique_candidates),
            "raw_boxes": raw_boxes,
            "candidates": candidates[:10],
        }

        output = []
        for candidate in accepted[: max(0, int(limite))]:
            x1, y1, x2, y2 = candidate["box"]
            output.append(
                (
                    float(candidate["score"]),
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                )
            )
        return output


def desenhar_candidatos(img_colorida, candidatos, cor):
    """Gera o debug visual no mesmo formato das funções PID originais."""
    if img_colorida is None:
        return None
    img_debug = img_colorida.copy()
    for indice, candidato in enumerate(candidatos):
        score, x, y, largura, altura = candidato
        cv2.rectangle(
            img_debug, (x, y), (x + largura, y + altura), cor, 1
        )
        text_y = max(12, y - 5)
        cv2.putText(
            img_debug,
            f"Teste #{indice + 1} ({score:.2f})",
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cor,
            1,
        )
    return img_debug
