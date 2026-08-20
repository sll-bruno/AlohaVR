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
SPECTATOR_CAMERAS = ("teleoperator_pov", "collaborator_pov", "overhead_cam", "worms_eye_cam")
SPECTATOR_WINDOW_NAME = "AlohaVR — visão do publico"


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
             anchored: bool, spectator_camera: str):
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

    # Diagnóstico de conexão: sem estes logs, uma falha de ICE aparece só como
    # "tela branca" no headset, sem nenhuma pista do que aconteceu.
    @headset.pc.on("iceconnectionstatechange")
    async def _on_ice_state():
        print(f"[ICE] {headset.pc.iceConnectionState}")

    @headset.pc.on("connectionstatechange")
    async def _on_conn_state():
        print(f"[PC] {headset.pc.connectionState}")

    headset.run_in_thread()

    headset_control = AnchoredControl() if anchored else HeadsetControl()
    headset_control.reset()

    ts, _info = env.reset()
    action = np.concatenate([
        ts["poses"]["left"],
        np.array([0.0]),
        ts["poses"]["right"],
        np.array([0.0]),
        ts["poses"]["middle"],
    ])
    env.step(action)

    print("Pronto. Segure o botão A (RButtonOne) no Quest para engatar o controle.")

    remote_sdp_reported = False
    frame_idx = 0
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
                print("[sim] física divergiu, resetando a cena...")
                send_popup_message(headset, "Simulação instável. Reiniciando a cena...", 1.5)
                ts, _info = env.reset()
                action = np.concatenate([
                    ts["poses"]["left"], np.array([0.0]),
                    ts["poses"]["right"], np.array([0.0]),
                    ts["poses"]["middle"],
                ])
                headset_control.reset()
                continue

            if terminated:
                send_popup_message(headset, f"Simulação terminou: {info}. Reiniciando...", 2.0)
                ts, _info = env.reset()
                headset_control.reset()
                continue

            if not remote_sdp_reported and headset.pc.remoteDescription is not None:
                remote_sdp_reported = True
                cands = [
                    l for l in headset.pc.remoteDescription.sdp.splitlines()
                    if l.startswith("a=candidate")
                ]
                print(f"[ICE] answer do Quest trouxe {len(cands)} candidato(s):")
                for c in cands:
                    print(f"      {c}")

            headset_data = headset.receive_data()
            feedback = HeadsetFeedback()
            feedback.info = "Segure o botão A para engatar o controle."

            if headset_data is not None:
                new_action, feedback = headset_control.run(
                    headset_data,
                    ts["poses"]["left"],
                    ts["poses"]["right"],
                    ts["poses"]["middle"],
                )

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

                if headset_control.is_running() and not headset_data.r_button_one:
                    # soltou o A: desengata, mantém a última pose
                    headset_control.reset()

            headset.send_feedback(feedback)

            # vídeo estéreo pro Quest
            left_img = render(env, EYE_CAMERAS[0], eye_width, eye_height)
            right_img = render(env, EYE_CAMERAS[1], eye_width, eye_height)
            headset.send_images(left_img, right_img)

            # vídeo de terceira pessoa pro público (mais barato: 1 a cada N frames)
            if show_spectator_window and frame_idx % spectator_every == 0:
                spectator_frame = cv2.cvtColor(
                    render(env, spectator_camera, 640, 360), cv2.COLOR_RGB2BGR
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
    parser.add_argument("--spectator-cam", dest="spectator_camera",
                        choices=SPECTATOR_CAMERAS, default="teleoperator_pov",
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
             args.spectator_camera)
