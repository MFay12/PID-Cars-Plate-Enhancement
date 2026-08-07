from DetectorHeuristico import DetectorHeuristicoPlaca, desenhar_candidatos


_DETECTOR = DetectorHeuristicoPlaca()


def buscar_placa_moto(img_original, img_colorida):
    """Busca placas de motocicleta mantendo o contrato original do PID."""
    top_5 = _DETECTOR.buscar(img_original, familias=("moto",), limite=5)
    img_debug = desenhar_candidatos(
        img_colorida, top_5, cor=(255, 165, 0)
    )
    return img_debug, top_5
