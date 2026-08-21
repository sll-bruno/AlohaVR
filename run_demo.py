"""
AlohaVR — loop de teleoperação ao vivo do robô simulado, sem gravação.

Diferença em relação a av-aloha/data_collection_scripts/record_sim_episodes.py:
este script NÃO grava episódios em HDF5 e roda em loop contínuo, sem limite de
passos — é feito para demonstração ao vivo (UPA 2026), não coleta de dados.

Pré-requisitos (ver README.md deste repositório):
  - av-aloha clonado ao lado deste repositório, com as dependências do
    requirements.txt instaladas num venv (recomendado Python 3.11+, por causa
    da wheel pré-compilada do pacote `av`).
  - serviceAccountKey.json e signalingSettings.json em
    av-aloha/data_collection_scripts/ (nunca commitados).
  - App instalado no Quest (TwoStreamGuidedVision.apk), Project ID/Password
    preenchidos e "Connect" pressionado na cena de teleop.

Uso:
  cd av-aloha/data_collection_scripts
  ../venv/bin/python /caminho/para/AlohaVR/run_demo.py --task sim_insert_peg
"""

import argparse
import asyncio
import os
import subprocess
import sys
import threading
import time

import cv2
import mujoco
import numpy as np

# Este script roda com av-aloha/data_collection_scripts no PYTHONPATH
# (ou a partir de dentro dessa pasta — ver docstring acima).
try:
    import sim_env as sim_env_module
    from sim_env import make_sim_env
    from webrtc_headset import WebRTCHeadset
    # HeadsetFullControl move os dois braços via VR (não só o braço do meio/cabeça)
    from headset_control import HeadsetFullControl as HeadsetControl
    from headset_utils import HeadsetFeedback, convert_right_to_left_coordinates
    from transform_utils import (
        align_rotation_to_z_axis, mat2pose, pose2mat, quat2mat,
        transform_coordinates, within_pose_threshold, wxyz_to_xyzw, xyzw_to_wxyz,
    )
    from constants import SIM_DT, SIM_PHYSICS_ENV_STEP_RATIO, SIM_TASK_CONFIGS
    from dm_control.rl.control import PhysicsError
except ImportError as e:
    sys.exit(
        "Não consegui importar os módulos do av-aloha. Rode este script a partir\n"
        "de dentro de av-aloha/data_collection_scripts, ou adicione essa pasta ao\n"
        f"PYTHONPATH. Erro original: {e}"
    )

# Renderizamos as câmeras nós mesmos (env criado com cameras=[]) para controlar
# resolução e proporção. O sim_env renderiza a zed_cam em 720x720 por olho —
# quadrado — mas o app Unity foi feito para a ZED real, 16:9; a imagem quadrada
# chega esticada e a perspectiva fica errada. Além disso 720x720 x2 é caro.
EYE_CAMERAS = ("zed_cam_left", "zed_cam_right")
ARM_LINK_KEYWORDS = ("shoulder", "upper_arm", "forearm", "wrist", "base_link")
# Vista frontal: além de enquadrar melhor, é nela que o público vê a cabeça do
# robô girar conforme quem está de headset move a própria cabeça — que é o ponto
# do projeto (visão ativa) e some numa vista de cima.
SPECTATOR_CAMERAS = ("collaborator_pov", "teleoperator_pov", "overhead_cam", "worms_eye_cam")
SPECTATOR_WINDOW_NAME = "AlohaVR"

# Mostrado dentro do headset e na tela do público. O público é de ensino médio e
# ninguém vai explicar dentro do óculos, então cada cena precisa dizer o que
# fazer em uma linha, sem jargão.
TASK_INSTRUCTIONS = {
    "sim_insert_peg": "Encaixe a peça vermelha no bloco azul",
    "sim_slot_insertion": "Encaixe o bastão verde na fenda rosa",
    "sim_sew_needle": "Passe a agulha verde pelo furo da parede",
    "sim_tube_transfer": "Leve a bolinha de um tubo para o outro",
    "sim_hook_package": "Pendure a caixa vermelha no gancho",
}
# Par de geoms que se tocam quando a tarefa é concluída. NÃO dá para usar o
# get_reward() do av-aloha: os marcadores têm gap="100", o que faz o MuJoCo
# listar contatos a até 100 metros (medido: pin <-> geom a 11 m), e o
# get_reward() só checa se o par aparece na lista, sem olhar distância — ele
# retorna "sucesso" com a peça a 26 cm do furo. Aqui filtramos por penetração
# real (dist < 0).
# "pair" confirma alinhamento lateral (a peça está dentro do furo); "seat"
# exige profundidade. Sem o segundo, encostar a ponta já contava como sucesso:
# o contato reporta sempre -2 cm, que é a penetração lateral, não o quanto
# entrou — a profundidade só sai da geometria dos corpos.
# O `pin` do insert_peg tem a MESMA seção da peça (2x2 cm) e fica atravessado no
# meio do tubo: medido, a peça começa a penetrá-lo a 2 cm de profundidade e trava
# ali. Ele deveria ser só sensor (gap="100" no XML), mas na prática barra. Por
# isso a colisão dele é desligada e o sucesso passa a sair da geometria dos
# corpos, sem depender de contato nenhum.
SUCCESS_SPECS = {
    "sim_insert_peg": {
        "geometric": {"moving": "peg", "socket": "hole",
                      "max_offset": 0.05, "max_lateral": 0.02},
        "disable_markers": ("pin",),
    },
    "sim_sew_needle": {"pair": ("pin-needle", "pin-wall")},
}
# afplay e os sons são nativos do macOS: nada para instalar nem versionar.
# O áudio sai só nas caixas do Mac — o app Unity ignora tracks que não sejam
# de vídeo (OnTrack filtra TrackKind.Video), então não há como levá-lo ao
# headset sem recompilar o APK. Serve de sinal para a plateia e o monitor.
SUCCESS_SOUND = "/System/Library/Sounds/Hero.aiff"
CONTROL_TABLE_HEADERS = ("Controle", "Ação")
CONTROL_TABLE_ROWS = (
    ("Segure A", "assumir o controle do robô"),
    ("Gatilhos", "abrir e fechar as garras"),
    ("Segure X", "recomeçar do zero"),
    ("B", "voltar à tela inicial"),
)
SUCCESS_MESSAGE = "Parabéns! Você concluiu a tarefa"
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


class AnchoredControl:
    """Controle em que cada membro é relativo à própria pose no engate.

    O HeadsetFullControl do av-aloha mapeia as mãos relativas à *cabeça*: a
    posição do controle em relação ao seu crânio vira a posição do braço em
    relação à câmera do robô. Na prática, engatar com as mãos na cintura joga
    os braços do robô para baixo na hora. Aqui cada membro guarda seu próprio
    par (pose sua, pose do robô) no instante do engate, então nada salta: o
    movimento passa a ser puramente incremental a partir dali.
    """

    def __init__(self, head_position_threshold=0.05, head_rotation_threshold=0.3,
                 motion_scale=1.0):
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
        # Amplia o deslocamento das mãos (não o da cabeça, que precisa ser 1:1
        # para a imagem não brigar com o sistema vestibular). Com 1.5, mover a
        # mão 10 cm move o braço 15 cm: cobre mais espaço de trabalho sem exigir
        # que a pessoa estique tanto o braço, ao custo de precisão fina.
        self.motion_scale = motion_scale
        self.reset()

    def reset(self):
        self.started = False
        self.anchors = {}

    def is_running(self):
        return self.started

    @staticmethod
    def _arm_mat(pose):
        return pose2mat(pose[:3], wxyz_to_xyzw(pose[3:]))

    def start(self, headset_data, left_arm_pose, right_arm_pose, middle_arm_pose):
        head = np.eye(4)
        head[:3, :3] = align_rotation_to_z_axis(quat2mat(headset_data.h_quat))
        head[:3, 3] = headset_data.h_pos

        middle = np.eye(4)
        middle[:3, :3] = align_rotation_to_z_axis(
            quat2mat(wxyz_to_xyzw(middle_arm_pose[3:]))
        )
        middle[:3, 3] = middle_arm_pose[:3]

        self.anchors = {
            "middle": (head, middle),
            "left": (pose2mat(headset_data.l_pos, headset_data.l_quat),
                     self._arm_mat(left_arm_pose)),
            "right": (pose2mat(headset_data.r_pos, headset_data.r_quat),
                      self._arm_mat(right_arm_pose)),
        }
        self.started = True

    def run(self, headset_data, left_arm_pose, right_arm_pose, middle_arm_pose):
        current = {
            "left": self._arm_mat(left_arm_pose),
            "right": self._arm_mat(right_arm_pose),
            "middle": self._arm_mat(middle_arm_pose),
        }
        user_now = {
            "left": pose2mat(headset_data.l_pos, headset_data.l_quat),
            "right": pose2mat(headset_data.r_pos, headset_data.r_quat),
            "middle": pose2mat(headset_data.h_pos, headset_data.h_quat),
        }

        if self.started:
            targets = {}
            for name in current:
                user_start, arm_start = self.anchors[name]
                alvo = transform_coordinates(user_now[name], user_start, arm_start)
                if name != "middle" and self.motion_scale != 1.0:
                    deslocamento = alvo[:3, 3] - arm_start[:3, 3]
                    alvo = alvo.copy()
                    alvo[:3, 3] = arm_start[:3, 3] + deslocamento * self.motion_scale
                targets[name] = alvo
            head_ref, middle_ref = self.anchors["middle"]
        else:
            # Antes de engatar o robô fica parado: alvo é a pose atual.
            targets = dict(current)
            head_ref, middle_ref = np.eye(4), current["middle"]

        parts = []
        for name, gripper in (("left", headset_data.l_index_trigger),
                              ("right", headset_data.r_index_trigger)):
            pos, quat = mat2pose(targets[name])
            parts += [pos, xyzw_to_wxyz(quat), np.array([gripper])]
        mid_pos, mid_quat = mat2pose(targets["middle"])
        parts += [mid_pos, xyzw_to_wxyz(mid_quat)]

        feedback = HeadsetFeedback()
        feedback.info = ""
        feedback.head_out_of_sync = not within_pose_threshold(
            current["middle"][:3, 3], current["middle"][:3, :3],
            targets["middle"][:3, 3], targets["middle"][:3, :3],
            self.head_position_threshold, self.head_rotation_threshold,
        )
        feedback.left_out_of_sync = False
        feedback.right_out_of_sync = False
        for name in ("left", "right", "middle"):
            in_head_frame = transform_coordinates(current[name], middle_ref, head_ref)
            pos, quat = convert_right_to_left_coordinates(*mat2pose(in_head_frame))
            setattr(feedback, f"{name}_arm_position", pos)
            setattr(feedback, f"{name}_arm_rotation", quat)

        return np.concatenate(parts), feedback


def widen_peg_hole(model, margin: float) -> bool:
    """Afasta as quatro placas que formam o tubo da cena sim_insert_peg.

    A abertura é de 3,6 x 3,6 cm para uma peça de 2 x 2 cm — 8 mm de folga por
    lado. Acertar 8 mm com a latência da demo, segurando os dois objetos no ar,
    é o que trava quem tem dois minutos: as garras chegam perto e a peça não
    entra. Só as placas se movem; o `pin`, que marca o sucesso, fica no centro
    e não é tocado.
    """
    # (geom, eixo em que se afasta, sentido, eixo em que precisa crescer)
    placas = (("hole-1", 2, -1, 1), ("hole-2", 2, +1, 1),   # piso e teto
              ("hole-3", 1, +1, 2), ("hole-4", 1, -1, 2))   # laterais
    try:
        ids = {n: model.name2id(n, "geom") for n, _, _, _ in placas}
    except Exception:
        return False
    for nome, eixo, sentido, eixo_cresce in placas:
        i = ids[nome]
        pos = np.array(model.geom_pos[i])
        size = np.array(model.geom_size[i])
        pos[eixo] += sentido * margin
        # As placas se encostavam exatamente nos cantos; afastá-las sem crescer
        # abre uma fenda visível do tamanho da margem. Crescer na direção
        # perpendicular mantém o tubo fechado.
        size[eixo_cresce] += margin
        model.geom_pos[i], model.geom_size[i] = pos, size
    return True


def set_socket_mass(model, body_name: str, grams: float) -> bool:
    """Ajusta a massa do alvo do encaixe.

    O bloco azul tem 101 g de origem contra 48 g da peça: encostar nele o
    empurra, e a pessoa passa a perseguir um alvo móvel enquanto mira. Pesar
    mais o deixa firme na mesa — mas peso demais faz a cena parecer travada,
    então é um número para calibrar, não para maximizar.
    """
    try:
        body = model.name2id(body_name, "body")
    except Exception:
        return False
    atual = float(model.body_mass[body])
    if atual <= 0:
        return False
    alvo = grams / 1000.0
    model.body_inertia[body] = np.array(model.body_inertia[body]) * (alvo / atual)
    model.body_mass[body] = alvo
    return True


def widen_needle_hole(model, margin: float) -> bool:
    """Afasta as bordas do vão da parede na cena sim_sew_needle.

    A parede é montada com caixas: dois pilares laterais (wall-2/3) e dois
    blocos centrais, embaixo e em cima (wall-4/5). O vão entre eles é de 3x3 cm
    e a agulha tem 2x2 cm — 5 mm de folga de cada lado, o que com a latência da
    demo torna a tarefa quase impossível para quem tem dois minutos.

    Só as bordas se movem: os marcadores pin-wall/pin-needle, que é o que o
    get_reward() usa para detectar a agulha enfiada, ficam no centro do vão e
    não são tocados.
    """
    try:
        ids = {n: model.name2id(n, "geom") for n in
               ("wall-2", "wall-3", "wall-4", "wall-5")}
    except Exception:
        return False  # cena sem parede: nada a fazer

    # Pilares recuam em y, mantendo a borda externa no lugar.
    for name, side in (("wall-2", +1), ("wall-3", -1)):
        i = ids[name]
        pos, size = np.array(model.geom_pos[i]), np.array(model.geom_size[i])
        outer = side * pos[1] + size[1]
        inner = side * pos[1] - size[1] + margin
        size[1] = (outer - inner) / 2
        pos[1] = side * (inner + size[1])
        model.geom_pos[i], model.geom_size[i] = pos, size

    # Blocos centrais acompanham a nova largura e recuam em z, cada um afastando
    # a face voltada para o vão e mantendo a outra.
    for name, moves_top in (("wall-4", True), ("wall-5", False)):
        i = ids[name]
        pos, size = np.array(model.geom_pos[i]), np.array(model.geom_size[i])
        size[1] += margin
        low, high = pos[2] - size[2], pos[2] + size[2]
        if moves_top:
            high -= margin
        else:
            low += margin
        size[2] = (high - low) / 2
        pos[2] = low + size[2]
        model.geom_pos[i], model.geom_size[i] = pos, size
    return True


# Ponto do mundo onde o texto é "pintado", atrás da bancada. Fixo no mundo:
# um texto colado na tela acompanha a cabeça enquanto a cena se move, e esse
# conflito é justamente o que embrulha o estômago em VR.
# z escolhido por medição: a 0.42 o texto sai pelo topo assim que a pessoa
# inclina a cabeça para a mesa — que é a postura natural ao manipular. A 0.28
# ele fica entre 13% e 25% da altura do quadro nas duas posturas.
WORLD_TEXT_ANCHOR = np.array([0.0, 0.36, 0.28])
# A rodinha fica logo acima da mensagem, no mesmo plano do mundo. 7 cm por
# medição: abaixo disso o anel encosta no texto (a 5 cm sobram 6 px).
WORLD_RING_ANCHOR = WORLD_TEXT_ANCHOR + np.array([0.0, 0.0, 0.07])
WORLD_RING_RADIUS = 0.045   # raio físico, em metros
WORLD_TEXT_HEIGHT = 0.055   # altura física do texto, em metros


def project_to_eye(model, data, cam_name, point, width, height):
    """Projeta um ponto do mundo no pixel correspondente de uma câmera.

    Como cada olho tem posição própria, o mesmo ponto cai em pixels diferentes
    nos dois — é essa disparidade que faz o texto parecer estar de fato lá no
    fundo, e não colado no rosto.
    """
    cam_id = model.name2id(cam_name, "camera")
    origin = np.array(data.cam_xpos[cam_id])
    rot = np.array(data.cam_xmat[cam_id]).reshape(3, 3)
    local = rot.T @ (np.asarray(point, dtype=float) - origin)
    depth = -local[2]           # a câmera do MuJoCo olha para -Z
    if depth <= 0.05:
        return None             # atrás da câmera ou colado nela
    focal = (height / 2) / np.tan(np.radians(float(model.cam_fovy[cam_id])) / 2)
    return (int(width / 2 + focal * local[0] / depth),
            int(height / 2 - focal * local[1] / depth),
            depth, focal)


def draw_hold_ring(frame, progress: float, center=None, radius=None):
    """Arco de progresso do gesto de segurar X.

    Sem isto o gesto é invisível: quem segura não tem sinal de que está
    acontecendo algo e conclui que o botão não funciona. Recebe centro e raio já
    projetados para ficar ancorado no mundo, junto da mensagem fixa — preso à
    tela ele acompanharia a cabeça, que é o que embrulha o estômago.
    """
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = int(min(h, w) * 0.09)
    center = (int(center[0]), int(center[1]))
    radius = max(6, int(radius))
    cv2.circle(frame, center, radius, (60, 60, 60), 6, cv2.LINE_AA)
    cv2.ellipse(frame, center, (radius, radius), -90, 0,
                360 * max(0.0, min(1.0, progress)), (90, 220, 120), 6, cv2.LINE_AA)
    return frame


class HeadsetText:
    """Texto acentuado dentro do headset, via sprite em cache.

    O cv2.putText só desenha ASCII ("peça" vira "pe?a") e converter o frame
    inteiro para PIL a cada frame, nos dois olhos, sairia caro. Aqui o texto é
    rasterizado uma vez por conteúdo e depois só composto sobre a imagem, que é
    uma operação de numpy.
    """

    def __init__(self):
        self.font = None
        self._cache = {}
        try:
            from PIL import ImageFont
            for path in FONT_CANDIDATES:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, 22)
                    break
        except ImportError:
            pass

    def _sprite(self, lines, color=None):
        key = (tuple(lines), color)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image, ImageDraw
        pad, line_h = 14, 30
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        width = max(int(probe.textlength(t, font=self.font)) for t in lines) + 2 * pad
        height = line_h * len(lines) + 2 * pad
        img = Image.new("RGBA", (width, height), color or (0, 0, 0, 150))
        draw = ImageDraw.Draw(img)
        for i, text in enumerate(lines):
            tw = int(probe.textlength(text, font=self.font))
            draw.text(((width - tw) // 2, pad + i * line_h), text,
                      font=self.font, fill=(255, 255, 255, 255))
        sprite = np.array(img)
        if len(self._cache) > 24:       # textos com contagem regressiva mudam muito
            self._cache.clear()
        self._cache[key] = sprite
        return sprite

    def table_sprite(self, headers, rows):
        """Cartão de controles em duas colunas, com cabeçalho."""
        key = ("__tabela__", headers, rows)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image, ImageDraw
        pad, line_h, col_gap = 22, 34, 34
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        col1 = max(int(probe.textlength(t, font=self.font))
                   for t in (headers[0], *(r[0] for r in rows)))
        col2 = max(int(probe.textlength(t, font=self.font))
                   for t in (headers[1], *(r[1] for r in rows)))
        width = col1 + col_gap + col2 + 2 * pad
        height = line_h * (len(rows) + 1) + 2 * pad + 10
        img = Image.new("RGBA", (width, height), (0, 0, 0, 165))
        draw = ImageDraw.Draw(img)
        draw.text((pad, pad), headers[0], font=self.font, fill=(150, 200, 255, 255))
        draw.text((pad + col1 + col_gap, pad), headers[1], font=self.font,
                  fill=(150, 200, 255, 255))
        rule_y = pad + line_h - 6
        draw.line([(pad, rule_y), (width - pad, rule_y)], fill=(120, 120, 120, 200))
        for i, (controle, acao) in enumerate(rows):
            y = pad + line_h * (i + 1) + 6
            draw.text((pad, y), controle, font=self.font, fill=(255, 255, 255, 255))
            draw.text((pad + col1 + col_gap, y), acao, font=self.font,
                      fill=(235, 235, 235, 255))
        sprite = np.array(img)
        self._cache[key] = sprite
        return sprite

    def blit(self, frame, sprite, cx, cy, scale=1.0):
        """Compõe um sprite centrado em (cx, cy), recortando o que sair da imagem."""
        if scale != 1.0:
            sh = max(1, int(sprite.shape[0] * scale))
            sw = max(1, int(sprite.shape[1] * scale))
            sprite = cv2.resize(sprite, (sw, sh), interpolation=cv2.INTER_AREA)
        sh, sw = sprite.shape[:2]
        h, w = frame.shape[:2]
        x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        sx1 = min(sw, sx0 + w - x0)
        sy1 = min(sh, sy0 + h - y0)
        if sx1 <= sx0 or sy1 <= sy0:
            return frame
        piece = sprite[sy0:sy1, sx0:sx1]
        alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
        region = frame[y0:y0 + piece.shape[0], x0:x0 + piece.shape[1]]
        frame[y0:y0 + piece.shape[0], x0:x0 + piece.shape[1]] = (
            region * (1 - alpha) + piece[:, :, 2::-1] * alpha
        ).astype(frame.dtype)
        return frame

    def draw_in_world(self, frame, model, data, cam_name, lines, point,
                      world_height=WORLD_TEXT_HEIGHT, color=None):
        """Texto ancorado num ponto do mundo, com perspectiva e paralaxe."""
        if self.font is None or not lines:
            return frame
        h, w = frame.shape[:2]
        proj = project_to_eye(model, data, cam_name, point, w, h)
        if proj is None:
            return frame
        cx, cy, depth, focal = proj
        sprite = self._sprite(tuple(lines), color)
        alvo_px = focal * world_height * len(lines) / depth
        return self.blit(frame, sprite, cx, cy, alvo_px / sprite.shape[0])

    def draw(self, frame, lines, at_center=False, y_offset=0, color=None):
        """Faixa inferior para o permanente, centro para o transitório.

        O painel do app ocupa 81,9° x 52°, então o extremo da imagem é
        desconfortável de ler — nada encostado na borda.
        """
        if self.font is None or not lines:
            return frame
        sprite = self._sprite(tuple(lines), color)
        sh, sw = sprite.shape[:2]
        h, w = frame.shape[:2]
        x = (w - sw) // 2
        # 14% da borda: no headset o extremo do painel cai na periferia da
        # visão, onde ler cansa. Longe da borda, e menor, lê-se melhor.
        y = ((h - sh) // 2 if at_center else h - sh - int(h * 0.14)) + y_offset
        if x < 0 or y < 0 or y + sh > h or x + sw > w:
            return frame
        alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
        region = frame[y:y + sh, x:x + sw]
        frame[y:y + sh, x:x + sw] = (
            region * (1 - alpha) + sprite[:, :, 2::-1] * alpha
        ).astype(frame.dtype)
        return frame


def disable_marker_collision(model, names) -> int:
    """Tira os marcadores de sucesso do caminho da física."""
    desligados = 0
    for nome in names:
        try:
            i = model.name2id(nome, "geom")
        except Exception:
            continue
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0
        desligados += 1
    return desligados


def task_completed(physics, spec) -> bool:
    """True quando a tarefa foi concluída de verdade, não só encostada."""
    if not spec:
        return False
    data, model = physics.data, physics.model

    geo = spec.get("geometric")
    if geo is not None:
        # Sem contato: a peça está encaixada quando o eixo dela coincide com o
        # do encaixe (alinhamento) e os centros estão próximos (profundidade).
        named = physics.named.data
        centro = np.array(named.xpos[geo["socket"]])
        eixo = np.array(named.xmat[geo["socket"]]).reshape(3, 3)[:, 0]
        delta = np.array(named.xpos[geo["moving"]]) - centro
        ao_longo = float(np.dot(delta, eixo))
        lateral = float(np.linalg.norm(delta - ao_longo * eixo))
        return (abs(ao_longo) <= geo["max_offset"]
                and lateral <= geo["max_lateral"])

    alvo = set(spec["pair"])
    tocando = False
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist >= 0:
            continue
        names = {
            mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if names == alvo:
            tocando = True
            break
    if not tocando:
        return False

    seat = spec.get("seat")
    if seat is None:
        return True
    # Quanto a peça avançou pelo eixo do furo: a diferença entre os centros,
    # projetada no eixo do encaixe. Perto de zero = assentada até o fundo.
    named = physics.named.data
    encaixe = np.array(named.xpos[seat["socket"]])
    eixo = np.array(named.xmat[seat["socket"]]).reshape(3, 3)[:, 0]
    desvio = abs(float(np.dot(np.array(named.xpos[seat["moving"]]) - encaixe, eixo)))
    return desvio <= seat["max_offset"]


def play_success_sound():
    """Toca o som de comemoração fora do caminho do frame.

    Medido: subprocess.Popen custa ~10 ms neste processo, o que é 20% de um
    frame e aparece como engasgo bem no momento da comemoração. Numa thread,
    o loop nem percebe.
    """
    if not os.path.exists(SUCCESS_SOUND):
        return

    def tocar():
        try:
            subprocess.run(["afplay", SUCCESS_SOUND],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except OSError:
            pass  # sem áudio a demo continua igual

    threading.Thread(target=tocar, daemon=True).start()


def neutral_action(ts) -> np.ndarray:
    """Ação que mantém cada braço onde está (garras abertas)."""
    return np.concatenate([
        ts["poses"]["left"], np.array([0.0]),
        ts["poses"]["right"], np.array([0.0]),
        ts["poses"]["middle"],
    ])


def reset_scene(env, headset_control):
    ts, _info = env.reset()
    headset_control.reset()
    return ts, neutral_action(ts)


class ConnectionWatchdog:
    """Republica o offer quando o headset cai, para o próximo usuário conectar.

    O próprio WebRTCHeadset do av-aloha já reinicia quando o ICE fecha
    (webrtc_headset.py, on_iceconnectionstatechange) — e na prática isso *dispara*
    quando alguém tira o óculos, contrariando a suposição inicial de que só
    "failed"/"disconnected" ocorreriam. O problema real, visto ao testar com o
    headset de verdade, foi outro: com as duas lógicas de restart rodando em
    paralelo, o watchdog às vezes derrubava uma conexão que o mecanismo original
    tinha acabado de restabelecer ("Connection closed, restarting..." aparecia
    de novo logo após "Data channel is open"). A correção é checar a *identidade*
    do objeto `pc`: se ele já mudou desde que o estado morto foi observado, o
    mecanismo original já agiu, e o watchdog não interfere.

    Mantido como rede de segurança para o caso em que o ICE nunca alcança
    "closed" de fato (ex: Wi-Fi cai sem fechamento limpo) — aí sim o watchdog é
    a única coisa que republica o offer.
    """

    DEAD_STATES = ("failed", "closed", "disconnected")

    def __init__(self, headset, grace_seconds=8.0, stuck_seconds=12.0):
        self.headset = headset
        self.grace_seconds = grace_seconds
        self.stuck_seconds = stuck_seconds
        self.connected = False
        self.dead_since = None
        self.pc_when_dead = None
        self.last_state = None
        self.last_pc = headset.pc
        self.answered_since = None

    def poll(self) -> bool:
        """Avança a máquina de estados. True quando um novo usuário conecta."""
        pc = self.headset.pc

        if pc is not self.last_pc:
            # O mecanismo original trocou o objeto `pc` (fechou e recriou) mais
            # rápido do que o nosso polling: um ciclo completo "closed" -> "new"
            # -> "connecting" -> "connected" pode acontecer entre duas chamadas
            # de poll(), e aí nunca observamos nenhum DEAD_STATE — self.connected
            # ficava preso em True e a cena não era reiniciada para quem chegou
            # depois. Identidade do objeto é um sinal confiável mesmo perdendo
            # os estados intermediários.
            self.last_pc = pc
            self.connected = False
            self.dead_since = None
            self.pc_when_dead = None
            self.answered_since = None
            self.last_state = None  # reimprime o estado do novo pc

        state = pc.connectionState
        if state != self.last_state:
            print(f"[webrtc] {state}")
            self.last_state = state

        if state == "connected":
            self.dead_since = None
            self.pc_when_dead = None
            self.answered_since = None
            if not self.connected:
                self.connected = True
                return True
            return False

        # Conexão que nunca completa: o app consumiu o offer (o run_offer só
        # segue adiante depois de receber a answer, e aí apaga o documento do
        # Firestore), mas o ICE não fecha. Nada republica, e o Load no headset
        # deixa de listar qualquer robô — visto ao vivo, e sem saída para quem
        # não pode reiniciar o processo no terminal.
        #
        # remoteDescription é o discriminador: em repouso, esperando o primeiro
        # usuário do dia, ele é None e o offer está corretamente parado no
        # Firestore. Só depois de uma answer consumida faz sentido cobrar prazo.
        if not self.connected and pc.remoteDescription is not None:
            now = time.time()
            if self.answered_since is None:
                self.answered_since = now
            elif now - self.answered_since >= self.stuck_seconds:
                self.answered_since = None
                print(f"[webrtc] preso em '{state}' após a answer; "
                      "republicando offer")
                asyncio.run_coroutine_threadsafe(
                    self.headset.restart_connection(), self.headset.event_loop
                )
            return False

        if state in self.DEAD_STATES and self.connected:
            # Carência: dá tempo do mecanismo original (iceConnectionState ==
            # "closed") agir primeiro, e "disconnected" às vezes se recupera
            # sozinho sem nenhum restart.
            now = time.time()
            if self.dead_since is None:
                self.dead_since = now
                self.pc_when_dead = pc
            elif now - self.dead_since >= self.grace_seconds:
                self.connected = False
                self.dead_since = None
                if self.headset.pc is self.pc_when_dead:
                    # Ninguém trocou o pc ainda: o mecanismo original não agiu
                    # (ou nunca vai agir, ex. queda de rede sem fechamento
                    # limpo). Republica por conta própria.
                    print("[webrtc] sem reação própria após a carência; "
                          "republicando offer para o próximo usuário")
                    asyncio.run_coroutine_threadsafe(
                        self.headset.restart_connection(), self.headset.event_loop
                    )
                else:
                    # O mecanismo original já trocou o pc por conta própria
                    # (ex: "Connection closed, restarting..."). Não faz nada —
                    # só volta a acompanhar o novo pc no próximo poll().
                    print("[webrtc] restart original já em andamento, "
                          "watchdog não interfere")
                self.pc_when_dead = None
        return False


class Overlay:
    """Texto sobre a janela do público. Usa Pillow para acentuar corretamente.

    O OpenCV só desenha fontes Hershey (ASCII), que transformam "peça" em "pe?a"
    numa tela que o público vai ficar olhando o dia todo.
    """

    def __init__(self):
        self.font = self.small = None
        try:
            from PIL import ImageFont
            for path in FONT_CANDIDATES:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, 26)
                    self.small = ImageFont.truetype(path, 18)
                    break
        except ImportError:
            pass  # sem Pillow o vídeo continua, só sem legenda

    def draw(self, frame_bgr, instruction: str, status: str, connected: bool,
             completions: int = 0, celebrating: bool = False, slow: bool = False):
        """O centro fica livre durante o uso — é onde o público olha o robô.

        Só é ocupado quando não há nada acontecendo (ninguém conectado) ou
        quando há algo a comemorar.
        """
        if self.font is None:
            return frame_bgr
        from PIL import Image, ImageDraw
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = img.size

        draw.rectangle([(0, h - 46), (w, h)], fill=(0, 0, 0, 170))
        draw.text((14, h - 38), instruction, font=self.font, fill=(255, 255, 255, 255))

        dot = (90, 220, 120) if connected else (235, 170, 60)
        draw.ellipse([(14, 16), (26, 28)], fill=dot)
        draw.text((34, 13), status, font=self.small, fill=(235, 235, 235, 255))

        if completions:
            texto = f"{completions} tarefas completas hoje"
            largura = int(draw.textlength(texto, font=self.small))
            draw.text((w - largura - 14, 13), texto, font=self.small,
                      fill=(235, 235, 235, 255))

        if slow:
            aviso = "desempenho baixo — feche outros aplicativos"
            largura = int(draw.textlength(aviso, font=self.small))
            draw.text(((w - largura) // 2, 13), aviso, font=self.small,
                      fill=(235, 170, 60, 255))

        if celebrating:
            self._centro(draw, w, h, "Conseguiu!", (90, 220, 120, 230))
        elif not connected:
            self._centro(draw, w, h, "Coloque o óculos para pilotar o robô",
                         (0, 0, 0, 190))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _centro(self, draw, w, h, texto, fundo):
        largura = int(draw.textlength(texto, font=self.font))
        pad = 26
        x0, y0 = (w - largura) // 2 - pad, h // 2 - 34
        draw.rectangle([(x0, y0), (x0 + largura + 2 * pad, y0 + 68)], fill=fundo)
        draw.text((x0 + pad, y0 + 20), texto, font=self.font,
                  fill=(255, 255, 255, 255))


def send_popup_message(headset: WebRTCHeadset, message: str, duration: float = 3.0):
    feedback = HeadsetFeedback()
    feedback.info = message
    headset.send_feedback(feedback)
    time.sleep(duration)


def boxify_collision_meshes(model, include_arm_links: bool) -> int:
    """Troca malhas de colisão pela sua caixa envolvente.

    Colisão malha-malha domina o mj_step (3.25 dos 3.29 ms). O modelo mantém
    geoms de visual e de colisão separados, então isto não muda nada do que
    aparece na tela. O `world` são 33 extrusões de alumínio alinhadas aos eixos
    (a gaiola em volta da bancada): a caixa é quase exata e o corpo é estático,
    então não afeta manipulação. Os elos dos braços são opcionais: a caixa
    envolvente de um elo diagonal fica bem maior que o elo e pode gerar colisão
    falsa, com o braço parecendo travar sem motivo.
    """
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    converted = 0
    for i in range(model.ngeom):
        if int(model.geom_type[i]) != mesh_type:
            continue
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        body = mujoco.mj_id2name(
            model._model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])
        ) or ""
        is_world = body == "world"
        is_arm_link = any(k in body for k in ARM_LINK_KEYWORDS) and "gripper" not in body
        # Os dedos das garras nunca entram aqui: é onde o agarre acontece.
        if not (is_world or (include_arm_links and is_arm_link)):
            continue

        mesh_id = int(model.geom_dataid[i])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = np.array(model.mesh_vert[start:start + count]).reshape(-1, 3)
        lo, hi = verts.min(axis=0), verts.max(axis=0)

        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array(model.geom_quat[i]))
        model.geom_pos[i] = np.array(model.geom_pos[i]) + rot.reshape(3, 3) @ ((lo + hi) / 2)
        model.geom_size[i][:3] = np.maximum((hi - lo) / 2, 1e-4)
        model.geom_type[i] = int(mujoco.mjtGeom.mjGEOM_BOX)
        converted += 1
    return converted


def render(env, camera_id: str, width: int, height: int) -> np.ndarray:
    """Frame contíguo da câmera pedida.

    O render do dm_control devolve uma view espelhada verticalmente, com stride
    negativo (-3840). O OpenCV não aceita stride negativo e falha com "Layout of
    the output array img is incompatible with cv::Mat" — o array é gravável, o
    problema é o layout. Normalizar aqui, e não em cada chamador, garante que
    qualquer overlay possa ser desenhado sobre o resultado.
    """
    frame = env._physics.render(height=height, width=width, camera_id=camera_id)
    return np.ascontiguousarray(frame)


def run_demo(task_name: str, show_spectator_window: bool, eye_width: int,
             eye_height: int, spectator_every: int, physics_timestep: float,
             fovy: float, multiccd: bool, substeps: int, collision: str,
             anchored: bool, spectator_camera: str, idle_reset: float,
             hole_margin: float, success_reset: float, motion_scale: float,
             socket_mass: float):
    if task_name not in SIM_TASK_CONFIGS:
        sys.exit(
            f"Task '{task_name}' não existe em SIM_TASK_CONFIGS. "
            f"Opções: {list(SIM_TASK_CONFIGS.keys())}"
        )

    # Substeps por frame: menos substeps com timestep maior cobre o mesmo tempo
    # simulado por menos CPU, ao custo de fidelidade de contato. sim_env.step lê
    # esta constante do módulo, por isso a troca é feita aqui.
    if substeps is not None:
        sim_env_module.SIM_PHYSICS_ENV_STEP_RATIO = substeps

    # cameras=[] : get_obs() não renderiza nada, nós cuidamos disso no loop.
    success_spec = SUCCESS_SPECS.get(task_name)
    print(f"Carregando ambiente '{task_name}'...")
    env = make_sim_env(task_name, cameras=[])

    # O env dá 20 substeps de física por frame. Com o timestep padrão (0.002) isso
    # são 40 ms de tempo simulado, mas nesta máquina custa ~100 ms de CPU: o robô
    # anda a ~0.4x da velocidade real, o que se sente como arrasto na teleop.
    # Aumentar o timestep faz cada frame cobrir mais tempo simulado, devolvendo
    # velocidade real ao custo de fidelidade do contato.
    model = env._physics.model
    model.opt.timestep = physics_timestep
    # Iterações do solver ficam no padrão do MuJoCo (100/50): reduzi-las custava
    # ~10 ms quando o frame era 105 ms, mas depois das otimizações de colisão o
    # ganho sumiu no ruído e o contato ficava pior — o bloco saía empurrado a
    # 0.59 m/s ao ser tocado, contra 0.28 m/s com o solver completo.

    # Perfilando o mj_step, 3.25 dos 3.29 ms ficam em colisão narrow-phase: o
    # MULTICCD emite ~5 pontos de contato por par convexo, chegando a ~340
    # contatos por passo. Desligá-lo cai para ~94 contatos e 2.85x mais rápido,
    # ao custo de agarre menos firme (menos pontos por par). --multiccd religa.
    if not multiccd:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)

    if hole_margin > 0:
        alargou = (widen_needle_hole(model, hole_margin / 1000.0)
                   or widen_peg_hole(model, hole_margin / 1000.0))
        if alargou:
            print(f"cena: encaixe alargado em {hole_margin:.0f} mm por borda")
    markers = (success_spec or {}).get("disable_markers", ())
    if markers and disable_marker_collision(model, markers):
        print(f"cena: colisão do marcador desligada ({', '.join(markers)}) — "
              "era ele que travava o encaixe")

    if socket_mass > 0 and task_name == "sim_insert_peg":
        if set_socket_mass(model, "hole", socket_mass):
            print(f"cena: alvo do encaixe com {socket_mass:.0f} g")

    if collision != "mesh":
        n = boxify_collision_meshes(model, include_arm_links=(collision == "all"))
        print(f"colisão: {n} malhas convertidas em caixas (modo '{collision}')")
    n_substeps = sim_env_module.SIM_PHYSICS_ENV_STEP_RATIO
    frame_period = physics_timestep * n_substeps

    # fovy é o FOV vertical. Renderizando 16:9 com o fovy=90 do XML, o FOV
    # horizontal viraria ~121° (grande-angular, imagem "esticada"). ~59° vertical
    # devolve ~90° horizontal, que é a geometria para a qual o app foi feito.
    for cam in EYE_CAMERAS:
        model.cam_fovy[model.name2id(cam, "camera")] = fovy

    print(f"física: timestep={physics_timestep}s x {n_substeps} substeps "
          f"-> {frame_period*1000:.0f} ms simulados por frame | olhos {eye_width}x{eye_height} fovy={fovy}")

    print("Iniciando WebRTC (aguardando conexão do Quest via Firestore)...")
    headset = WebRTCHeadset()

    headset.run_in_thread()

    # O watchdog também loga as transições de estado: handlers presos ao objeto
    # pc parariam de disparar depois de um restart, que troca o pc inteiro.
    watchdog = ConnectionWatchdog(headset)
    overlay = Overlay()
    hud = HeadsetText()
    if success_spec is None:
        print(f"aviso: sem detecção de sucesso para '{task_name}'")
    instruction = TASK_INSTRUCTIONS.get(task_name, "Use os controles para mover os braços")

    headset_control = (AnchoredControl(motion_scale=motion_scale)
                       if anchored else HeadsetControl())
    headset_control.reset()

    ts, action = reset_scene(env, headset_control)
    env.step(action)

    print(f"Pronto: \"{instruction}\". A engata o controle, segure X para reiniciar tudo.")

    frame_idx = 0
    hold_reset_start = None
    hold_reset_fired = False
    HOLD_RESET_SECONDS = 1.5
    CARD_SECONDS = 10.0
    completions = 0
    was_completed = False
    success_at = None
    session_started_at = None
    slow_windows = 0
    running_slow = False
    last_data_at = time.time()
    scene_is_fresh = True
    fps_frame0 = 0
    fps_t0 = time.time()

    try:
        while True:
            step_start = time.time()

            # Timestep maior que o padrão troca fidelidade por velocidade e pode
            # divergir sob contato (mjWARN_BADQACC). Numa demo ao vivo isso não
            # pode derrubar o processo: reseta a cena e segue.
            try:
                ts, _reward, terminated, _truncated, info = env.step(action)
            except PhysicsError:
                print("[sim] física divergiu, resetando a cena")
                ts, action = reset_scene(env, headset_control)
                scene_is_fresh = True
                continue

            if terminated:
                print(f"[sim] cena concluída ({info}), reiniciando")
                ts, action = reset_scene(env, headset_control)
                scene_is_fresh = True
                continue

            # Um novo usuário chegou: entrega a cena limpa, sem depender de
            # alguém mexer no computador entre um aluno e outro.
            if watchdog.poll():
                print("[sessão] novo usuário conectado, cena reiniciada")
                ts, action = reset_scene(env, headset_control)
                scene_is_fresh = True
                session_started_at = time.time()
                success_at = None
                was_completed = False

            headset_data = headset.receive_data()
            feedback = HeadsetFeedback()
            hold_progress = None

            if headset_data is not None:
                last_data_at = time.time()
                new_action, feedback = headset_control.run(
                    headset_data,
                    ts["poses"]["left"],
                    ts["poses"]["right"],
                    ts["poses"]["middle"],
                )

                # Segurar X (esquerdo) por HOLD_RESET_SECONDS reinicia tudo do
                # zero, incluindo a referência de cabeça/mãos — reset_scene já
                # chama headset_control.reset(), que limpa as âncoras da
                # AnchoredControl; a próxima vez que A for apertado, a referência
                # é recalculada na pose atual da pessoa. Toque rápido não faz
                # nada, para evitar reinício sem querer.
                if headset_data.l_button_one:
                    if hold_reset_start is None:
                        hold_reset_start = time.time()
                    held_for = time.time() - hold_reset_start
                    if held_for >= HOLD_RESET_SECONDS and not hold_reset_fired:
                        print("[sessão] reinício completo pedido (X segurado)")
                        ts, action = reset_scene(env, headset_control)
                        scene_is_fresh = True
                        hold_reset_fired = True
                        was_completed = False
                        success_at = None
                        feedback.info = "Reiniciado!"
                        headset.send_feedback(feedback)
                        continue
                    elif not hold_reset_fired:
                        hold_progress = held_for / HOLD_RESET_SECONDS
                else:
                    hold_reset_start = None
                    hold_reset_fired = False

                if not headset_control.is_running() and headset_data.r_button_one:
                    # Apertar A é o sinal de "entendi": some com o cartão de
                    # controles sem prender quem já sabe pelos 10 s inteiros.
                    session_started_at = None
                    if anchored:
                        headset_control.start(
                            headset_data, ts["poses"]["left"],
                            ts["poses"]["right"], ts["poses"]["middle"],
                        )
                    else:
                        headset_control.start(headset_data, ts["poses"]["middle"])

                if headset_control.is_running():
                    action = new_action
                    scene_is_fresh = False

                if headset_control.is_running() and not headset_data.r_button_one:
                    # soltou o A: desengata, mantém a última pose
                    headset_control.reset()
            elif (not scene_is_fresh and idle_reset > 0
                  and time.time() - last_data_at > idle_reset):
                # NÃO detecta "tirou o headset": o WebRTCStreamer.cs só limpa
                # coisa nenhuma em OnDestroy (sair da PassthroughScene) — não
                # tem OnApplicationPause. Tirar o headset da cabeça apenas apaga
                # a tela; a pose dos controles continua chegando o tempo todo.
                # Isto só cobre o caso de a pose realmente parar de chegar por
                # outro motivo (app travou, processo do outro lado morreu sem
                # fechar a conexão) — não substitui alguém voltar pela tela
                # inicial entre um aluno e outro.
                print(f"[sessão] {idle_reset:.0f}s sem uso, cena reiniciada")
                ts, action = reset_scene(env, headset_control)
                scene_is_fresh = True

            # Sucesso: só na borda de subida, senão comemora todo frame.
            completed_now = task_completed(env._physics, success_spec)
            if completed_now and not was_completed:
                completions += 1
                success_at = time.time()
                play_success_sound()
                print(f"[sessão] tarefa concluída ({completions} hoje)")
            was_completed = completed_now

            celebrating = success_at is not None and time.time() - success_at < 6.0
            if (success_reset > 0 and success_at is not None
                    and time.time() - success_at >= success_reset):
                print("[sessão] reiniciando após o sucesso")
                ts, action = reset_scene(env, headset_control)
                scene_is_fresh = True
                success_at = None
                was_completed = False
                continue

            # O app desenha feedback.info no próprio infoText (WebRTCStreamer.cs:155),
            # o que duplicava a instrução: uma na UI do app e outra no vídeo. A do
            # vídeo é a que controlamos (posição, tamanho, acentos), então esta
            # fica vazia de propósito.
            feedback.info = ""
            headset.send_feedback(feedback)

            # Sem ninguém conectado não há para quem renderizar os olhos. São
            # ~25 ms por frame que, num MacBook Air sem ventoinha e 8 h de
            # evento, viram calor que degrada quem vem no próximo grupo.
            if watchdog.connected:
                left_img = render(env, EYE_CAMERAS[0], eye_width, eye_height)
                right_img = render(env, EYE_CAMERAS[1], eye_width, eye_height)

                showing_card = (session_started_at is not None
                                and time.time() - session_started_at < CARD_SECONDS)
                card = hud.table_sprite(CONTROL_TABLE_HEADERS, CONTROL_TABLE_ROWS)
                for eye, cam in zip((left_img, right_img), EYE_CAMERAS):
                    # Instrução e parabéns ficam presos ao mundo, "pintados" ao
                    # fundo da bancada: texto colado na tela acompanha a cabeça
                    # enquanto a cena se move, e é esse conflito que enjoa.
                    hud.draw_in_world(
                        eye, model, env._physics.data, cam,
                        ["Segure para reiniciar"] if hold_progress is not None
                        else [instruction],
                        WORLD_TEXT_ANCHOR,
                    )
                    # O parabéns é a exceção que fica preso à tela: quem acabou
                    # de encaixar está olhando para a mesa, e ancorado no mundo
                    # ele ficaria fora de vista justo na hora que importa. Dura
                    # poucos segundos, então não é o tipo de elemento estático
                    # que provoca enjoo.
                    if celebrating:
                        hud.draw(eye, [SUCCESS_MESSAGE], at_center=True,
                                 color=(20, 90, 40, 200))
                    # Estes são momentâneos, então seguir a cabeça é aceitável.
                    if hold_progress is not None:
                        alvo = project_to_eye(model, env._physics.data, cam,
                                              WORLD_RING_ANCHOR,
                                              eye.shape[1], eye.shape[0])
                        if alvo is not None:
                            cx, cy, depth, focal = alvo
                            draw_hold_ring(eye, hold_progress, (cx, cy),
                                           focal * WORLD_RING_RADIUS / depth)
                    elif showing_card:
                        h, w = eye.shape[:2]
                        hud.blit(eye, card, w // 2, int(h * 0.42))

                headset.send_images(left_img, right_img)

            # vídeo de terceira pessoa pro público (mais barato: 1 a cada N frames)
            if show_spectator_window and frame_idx % spectator_every == 0:
                spectator_frame = cv2.cvtColor(
                    render(env, spectator_camera, 960, 540), cv2.COLOR_RGB2BGR
                )
                spectator_frame = overlay.draw(
                    spectator_frame, instruction,
                    "Óculos conectado" if watchdog.connected else "Aguardando óculos",
                    watchdog.connected, completions, celebrating, running_slow,
                )
                cv2.imshow(SPECTATOR_WINDOW_NAME, spectator_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if time.time() - fps_t0 >= 3.0:
                hz = (frame_idx - fps_frame0) / 3.0
                alvo = 1 / frame_period
                print(f"[perf] {frame_idx - fps_frame0} frames em 3s -> "
                      f"{hz:.1f} Hz (alvo {alvo:.0f})")
                # Só conta quando há alguém conectado: em repouso pulamos os
                # renders de propósito e a taxa cai por decisão nossa.
                if watchdog.connected and hz < alvo * 0.6:
                    slow_windows += 1
                else:
                    slow_windows = 0
                if slow_windows == 2 and not running_slow:
                    running_slow = True
                    print("[perf] desempenho baixo sustentado — algum outro "
                          "aplicativo disputando CPU/GPU?")
                elif slow_windows == 0:
                    running_slow = False
                fps_t0, fps_frame0 = time.time(), frame_idx

            time_until_next_step = frame_period - (time.time() - step_start)
            time.sleep(max(0, time_until_next_step))

    except KeyboardInterrupt:
        print("\nEncerrando demo...")
    finally:
        if show_spectator_window:
            cv2.destroyAllWindows()
        headset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlohaVR — demo de teleop ao vivo")
    parser.add_argument(
        "--task", dest="task_name", type=str, default="sim_insert_peg",
        help="Nome da task do av-aloha (ver constants.SIM_TASK_CONFIGS)",
    )
    parser.add_argument(
        "--no-spectator-window", dest="spectator", action="store_false",
        help="Desativa a janela de terceira pessoa (tela do público)",
    )
    parser.add_argument("--eye-width", type=int, default=1280,
                        help="Largura de cada olho (padrão 1280 = resolução nativa do painel do app)")
    parser.add_argument("--eye-height", type=int, default=720, help="Altura de cada olho")
    parser.add_argument("--spectator-every", type=int, default=5,
                        help="Renderiza a janela do público 1 a cada N frames")
    parser.add_argument("--physics-timestep", type=float, default=0.0025,
                        help="Timestep da física. 0.002=fiel mas 0.4x tempo real aqui; "
                             "0.004=~0.75x e estável; 0.005=tempo real porém diverge sob contato")
    parser.add_argument("--fovy", type=float, default=70.0,
                        help="FOV vertical dos olhos. 52 é a geometria exata do painel do app; 70 abre o campo de visão e foi o valor validado no headset")
    parser.add_argument("--socket-mass", type=float, default=160.0,
                        help="Massa em gramas do alvo do encaixe. O original tem 101 g "
                             "e foge ao ser tocado; peso demais deixa a cena travada. "
                             "0 mantém o valor original")
    parser.add_argument("--motion-scale", type=float, default=1.0,
                        help="Multiplica o deslocamento das mãos (1.5 = mover a mão "
                             "10 cm move o braço 15 cm). Ajuda a alcançar sem esticar "
                             "o braço, ao custo de precisão fina. A cabeça segue 1:1")
    parser.add_argument("--success-reset", type=float, default=15.0,
                        help="Segundos após concluir a tarefa até reiniciar a cena "
                             "(0 mantém a cena como ficou)")
    parser.add_argument("--hole-margin", type=float, default=3.0,
                        help="Milímetros a afastar cada borda do encaixe (furo do "
                             "insert_peg ou vão da parede do sew_needle). 0 mantém a "
                             "geometria original, apertada demais para dois minutos")
    parser.add_argument("--idle-reset", type=float, default=25.0,
                        help="Segundos sem receber pose até reiniciar a cena "
                             "para o próximo usuário (0 desativa)")
    parser.add_argument("--spectator-cam", dest="spectator_camera",
                        choices=SPECTATOR_CAMERAS, default="collaborator_pov",
                        help="Câmera da tela do público")
    parser.add_argument("--head-relative-hands", dest="anchored", action="store_false",
                        help="Volta ao mapeamento do av-aloha: mãos relativas à cabeça "
                             "(os braços saltam para onde suas mãos estiverem ao engatar)")
    parser.add_argument("--collision", choices=("mesh", "world", "all"), default="world",
                        help="mesh=original; world=estrutura vira caixas (~21%% mais rápido, "
                             "sem efeito na manipulação); all=inclui elos dos braços "
                             "(~35%%, pode causar colisão falsa)")
    parser.add_argument("--substeps", type=int, default=None,
                        help="Substeps de física por frame (padrão 20, do av-aloha). "
                             "Menos substeps com timestep maior = mais rápido, menos fiel")
    parser.add_argument("--multiccd", action="store_true",
                        help="Religa o MULTICCD: agarre mais firme, ~2.8x mais lento")
    args = parser.parse_args()

    run_demo(args.task_name, args.spectator, args.eye_width, args.eye_height,
             args.spectator_every, args.physics_timestep, args.fovy, args.multiccd, args.substeps, args.collision, args.anchored,
             args.spectator_camera, args.idle_reset, args.hole_margin,
             args.success_reset, args.motion_scale, args.socket_mass)
