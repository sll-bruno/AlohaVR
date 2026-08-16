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
    from constants import SIM_DT, SIM_TASK_CONFIGS
except ImportError as e:
    sys.exit(
        "Não consegui importar os módulos do av-aloha. Rode este script a partir\n"
        "de dentro de av-aloha/data_collection_scripts, ou adicione essa pasta ao\n"
        f"PYTHONPATH. Erro original: {e}"
    )

SPECTATOR_CAMERA = "cam_high"  # câmera de terceira pessoa, pra tela externa
SPECTATOR_WINDOW_NAME = "AlohaVR — visão do publico"


def send_popup_message(headset: WebRTCHeadset, message: str, duration: float = 3.0):
    feedback = HeadsetFeedback()
    feedback.info = message
    headset.send_feedback(feedback)
    time.sleep(duration)


def split_stereo(zed_img: np.ndarray):
    """zed_cam vem como um único frame com os dois olhos lado a lado."""
    half = zed_img.shape[1] // 2
    left_img = zed_img[:, :half, :]
    right_img = zed_img[:, half:, :]
    return left_img, right_img


def run_demo(task_name: str, show_spectator_window: bool):
    if task_name not in SIM_TASK_CONFIGS:
        sys.exit(
            f"Task '{task_name}' não existe em SIM_TASK_CONFIGS. "
            f"Opções: {list(SIM_TASK_CONFIGS.keys())}"
        )

    cameras = ["zed_cam"]
    if show_spectator_window:
        cameras.append(SPECTATOR_CAMERA)

    print(f"Carregando ambiente '{task_name}' com câmeras {cameras}...")
    env = make_sim_env(task_name, cameras=cameras)

    print("Iniciando WebRTC (aguardando conexão do Quest via Firestore)...")
    headset = WebRTCHeadset()
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

    print("Pronto. Segure o gatilho direito (RButtonOne) no Quest para engatar o controle.")

    try:
        while True:
            step_start = time.time()

            ts, _reward, terminated, _truncated, info = env.step(action)

            if terminated:
                send_popup_message(headset, f"Simulação terminou: {info}. Reiniciando...", 2.0)
                ts, _info = env.reset()
                headset_control.reset()
                continue

            headset_data = headset.receive_data()
            feedback = HeadsetFeedback()
            feedback.info = "Segure o gatilho direito para engatar o controle."

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
                    # soltou o gatilho: desengata, mantém a última pose
                    headset_control.reset()

            headset.send_feedback(feedback)

            # vídeo estéreo pro Quest
            left_img, right_img = split_stereo(ts["images"]["zed_cam"])
            headset.send_images(left_img, right_img)

            # vídeo de terceira pessoa pro público
            if show_spectator_window:
                spectator_frame = cv2.cvtColor(ts["images"][SPECTATOR_CAMERA], cv2.COLOR_RGB2BGR)
                cv2.imshow(SPECTATOR_WINDOW_NAME, spectator_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time_until_next_step = SIM_DT - (time.time() - step_start)
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
    args = parser.parse_args()

    run_demo(args.task_name, args.spectator)
