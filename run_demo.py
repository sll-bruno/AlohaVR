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
import numpy as np

# Este script roda com av-aloha/data_collection_scripts no PYTHONPATH
# (ou a partir de dentro dessa pasta — ver docstring acima).
try:
    from sim_env import make_sim_env
    from webrtc_headset import WebRTCHeadset
    # HeadsetFullControl move os dois braços via VR (não só o braço do meio/cabeça)
    from headset_control import HeadsetFullControl as HeadsetControl
    from headset_utils import HeadsetFeedback
    from constants import SIM_DT, SIM_PHYSICS_ENV_STEP_RATIO, SIM_TASK_CONFIGS
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
SPECTATOR_CAMERA = "overhead_cam"  # terceira pessoa, pra tela externa
SPECTATOR_WINDOW_NAME = "AlohaVR — visão do publico"


def send_popup_message(headset: WebRTCHeadset, message: str, duration: float = 3.0):
    feedback = HeadsetFeedback()
    feedback.info = message
    headset.send_feedback(feedback)
    time.sleep(duration)


def render(env, camera_id: str, width: int, height: int) -> np.ndarray:
    return env._physics.render(height=height, width=width, camera_id=camera_id)


def run_demo(task_name: str, show_spectator_window: bool, eye_width: int,
             eye_height: int, spectator_every: int, physics_timestep: float,
             fovy: float):
    if task_name not in SIM_TASK_CONFIGS:
        sys.exit(
            f"Task '{task_name}' não existe em SIM_TASK_CONFIGS. "
            f"Opções: {list(SIM_TASK_CONFIGS.keys())}"
        )

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
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    frame_period = physics_timestep * SIM_PHYSICS_ENV_STEP_RATIO

    # fovy é o FOV vertical. Renderizando 16:9 com o fovy=90 do XML, o FOV
    # horizontal viraria ~121° (grande-angular, imagem "esticada"). ~59° vertical
    # devolve ~90° horizontal, que é a geometria para a qual o app foi feito.
    for cam in EYE_CAMERAS:
        model.cam_fovy[model.name2id(cam, "camera")] = fovy

    print(f"física: timestep={physics_timestep}s x {SIM_PHYSICS_ENV_STEP_RATIO} substeps "
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

    headset_control = HeadsetControl()
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

            ts, _reward, terminated, _truncated, info = env.step(action)

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
                    render(env, SPECTATOR_CAMERA, 640, 360), cv2.COLOR_RGB2BGR
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
    parser.add_argument("--eye-width", type=int, default=640,
                        help="Largura de cada olho (padrão 640, 16:9 com 360)")
    parser.add_argument("--eye-height", type=int, default=360, help="Altura de cada olho")
    parser.add_argument("--spectator-every", type=int, default=3,
                        help="Renderiza a janela do público 1 a cada N frames")
    parser.add_argument("--physics-timestep", type=float, default=0.005,
                        help="Timestep da física (0.002=fiel porém câmera lenta aqui; "
                             "0.005=tempo real)")
    parser.add_argument("--fovy", type=float, default=59.0,
                        help="FOV vertical das câmeras dos olhos (59 ~= 90° horizontal em 16:9)")
    args = parser.parse_args()

    run_demo(args.task_name, args.spectator, args.eye_width, args.eye_height,
             args.spectator_every, args.physics_timestep, args.fovy)
