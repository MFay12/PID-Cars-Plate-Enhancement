from DetectorHeuristico import DetectorHeuristicoPlaca, desenhar_candidatos


_DETECTOR = DetectorHeuristicoPlaca()


def buscar_placa_carro(img_original, img_colorida):
    """Busca placas horizontais mantendo o contrato original do PID."""
    top_5 = _DETECTOR.buscar(img_original, familias=("carro",), limite=5)
    img_debug = desenhar_candidatos(
        img_colorida, top_5, cor=(0, 165, 255)
    )
    return img_debug, top_5
