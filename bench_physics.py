"""
Mede se uma máquina aguenta a demo do AlohaVR, sem precisar de headset nem rede.

O gargalo é o mj_step do MuJoCo, que é single-thread e depende de CPU por núcleo;
o render é barato e já roda em GPU. Este script separa os dois custos e projeta
a taxa que a demo alcançaria.

Uso (com o venv do av-aloha):
    cd av-aloha/data_collection_scripts
    PYTHONPATH=$PWD ../venv/bin/python /caminho/AlohaVR/bench_physics.py
"""
import platform
import time

import mujoco
import numpy as np

from constants import SIM_PHYSICS_ENV_STEP_RATIO
from sim_env import make_sim_env

# Espelha exatamente a configuração do run_demo.py: medir o modelo sem as
# otimizações daria um número que não corresponde ao que a demo roda.
from run_demo import boxify_collision_meshes

TIMESTEP = 0.0025
EYE_W, EYE_H = 1280, 720


def main():
    print(f"máquina: {platform.machine()} | {platform.processor() or platform.system()}")

    env = make_sim_env("sim_insert_peg", cameras=[])
    model = env._physics.model
    model.opt.timestep = TIMESTEP
    # Iterações do solver ficam no padrão do MuJoCo (100/50): reduzi-las custava
    # ~10 ms quando o frame era 105 ms, mas depois das otimizações de colisão o
    # ganho sumiu no ruído e o contato ficava pior — o bloco saía empurrado a
    # 0.59 m/s ao ser tocado, contra 0.28 m/s com o solver completo.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    converted = boxify_collision_meshes(model, include_arm_links=False)
    print(f"colisão: {converted} malhas estáticas convertidas em caixas")

    obs, _ = env.reset()
    action = np.concatenate([
        obs["poses"]["left"], [0.0], obs["poses"]["right"], [0.0], obs["poses"]["middle"],
    ])
    for _ in range(5):  # aquece JIT do numba e caches
        env.step(action)

    N = 30
    t0 = time.time()
    for _ in range(N):
        env.step(action)
    step_ms = (time.time() - t0) / N * 1000

    for _ in range(3):
        env._physics.render(height=EYE_H, width=EYE_W, camera_id="zed_cam_left")
    t0 = time.time()
    for _ in range(N):
        env._physics.render(height=EYE_H, width=EYE_W, camera_id="zed_cam_left")
        env._physics.render(height=EYE_H, width=EYE_W, camera_id="zed_cam_right")
    render_ms = (time.time() - t0) / N * 1000

    frame_ms = step_ms + render_ms
    sim_ms = TIMESTEP * SIM_PHYSICS_ENV_STEP_RATIO * 1000

    print(f"física + IK ({SIM_PHYSICS_ENV_STEP_RATIO} substeps): {step_ms:6.1f} ms")
    print(f"render 2 olhos {EYE_W}x{EYE_H}:              {render_ms:6.1f} ms")
    print(f"frame completo:                        {frame_ms:6.1f} ms -> {1000/frame_ms:.1f} Hz")
    print(f"velocidade vs tempo real:              {sim_ms/frame_ms:.2f}x  (1.0 = tempo real)")
    print()
    print("referência MacBook Air M-series (mesma config): ~20 ms física+IK, "
          "~28 ms render, ~16 Hz ao vivo")


if __name__ == "__main__":
    main()
