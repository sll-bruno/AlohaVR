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
import sys
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

    def __init__(self, head_position_threshold=0.05, head_rotation_threshold=0.3):
        self.head_position_threshold = head_position_threshold
        self.head_rotation_threshold = head_rotation_threshold
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
            targets = {
                name: transform_coordinates(user_now[name], *self.anchors[name])
                for name in current
            }
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

    def __init__(self, headset, grace_seconds=8.0):
        self.headset = headset
        self.grace_seconds = grace_seconds
        self.connected = False
        self.dead_since = None
        self.pc_when_dead = None
        self.last_state = None
        self.last_pc = headset.pc

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
            self.last_state = None  # reimprime o estado do novo pc

        state = pc.connectionState
        if state != self.last_state:
            print(f"[webrtc] {state}")
            self.last_state = state

        if state == "connected":
            self.dead_since = None
            self.pc_when_dead = None
            if not self.connected:
                self.connected = True
                return True
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

    def draw(self, frame_bgr, instruction: str, status: str, connected: bool):
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
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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
    return env._physics.render(height=height, width=width, camera_id=camera_id)


def run_demo(task_name: str, show_spectator_window: bool, eye_width: int,
             eye_height: int, spectator_every: int, physics_timestep: float,
             fovy: float, multiccd: bool, substeps: int, collision: str,
             anchored: bool, spectator_camera: str, idle_reset: float,
             hole_margin: float):
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

    if hole_margin > 0 and widen_needle_hole(model, hole_margin / 1000.0):
        print(f"cena: vão da parede alargado em {hole_margin:.0f} mm por borda")

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
    instruction = TASK_INSTRUCTIONS.get(task_name, "Use os controles para mover os braços")

    headset_control = AnchoredControl() if anchored else HeadsetControl()
    headset_control.reset()

    ts, action = reset_scene(env, headset_control)
    env.step(action)

    print(f"Pronto: \"{instruction}\". A engata o controle, segure X para reiniciar tudo.")

    frame_idx = 0
    hold_reset_start = None
    hold_reset_fired = False
    HOLD_RESET_SECONDS = 1.5
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

            headset_data = headset.receive_data()
            feedback = HeadsetFeedback()

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
                        feedback.info = "Reiniciado!"
                        headset.send_feedback(feedback)
                        continue
                    elif not hold_reset_fired:
                        remaining = HOLD_RESET_SECONDS - held_for
                        feedback.info = f"Segurando X para reiniciar... {remaining:.1f}s"
                else:
                    hold_reset_start = None
                    hold_reset_fired = False

                if not headset_control.is_running() and headset_data.r_button_one:
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

            if not feedback.info:  # não sobrescreve o aviso de "segurando X..."
                feedback.info = (
                    instruction if headset_control.is_running()
                    else f"Segure A para começar\n{instruction}"
                )
            headset.send_feedback(feedback)

            # vídeo estéreo pro Quest
            left_img = render(env, EYE_CAMERAS[0], eye_width, eye_height)
            right_img = render(env, EYE_CAMERAS[1], eye_width, eye_height)
            headset.send_images(left_img, right_img)

            # vídeo de terceira pessoa pro público (mais barato: 1 a cada N frames)
            if show_spectator_window and frame_idx % spectator_every == 0:
                spectator_frame = cv2.cvtColor(
                    render(env, spectator_camera, 960, 540), cv2.COLOR_RGB2BGR
                )
                spectator_frame = overlay.draw(
                    spectator_frame, instruction,
                    "Óculos conectado" if watchdog.connected else "Aguardando óculos",
                    watchdog.connected,
                )
                cv2.imshow(SPECTATOR_WINDOW_NAME, spectator_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if time.time() - fps_t0 >= 3.0:
                print(f"[perf] {frame_idx - fps_frame0} frames em 3s -> "
                      f"{(frame_idx - fps_frame0)/3.0:.1f} Hz (alvo {1/frame_period:.0f})")
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
    parser.add_argument("--spectator-every", type=int, default=3,
                        help="Renderiza a janela do público 1 a cada N frames")
    parser.add_argument("--physics-timestep", type=float, default=0.0025,
                        help="Timestep da física. 0.002=fiel mas 0.4x tempo real aqui; "
                             "0.004=~0.75x e estável; 0.005=tempo real porém diverge sob contato")
    parser.add_argument("--fovy", type=float, default=52.0,
                        help="FOV vertical dos olhos. 52 = ângulo real que o painel 1280x720 do app ocupa (1.73x0.98m a 1m)")
    parser.add_argument("--hole-margin", type=float, default=5.0,
                        help="Milímetros a afastar cada borda do vão em sim_sew_needle "
                             "(0 mantém o original, que é apertado demais para a demo)")
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
             args.spectator_camera, args.idle_reset, args.hole_margin)
