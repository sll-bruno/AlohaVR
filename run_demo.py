# AlohaVR — loop de teleoperação ao vivo do robô simulado, sem gravação.
#
# STATUS: esqueleto ainda não implementado. Ver README.md para o plano completo.
#
# Este script deve rodar de dentro de um ambiente com av-aloha instalado
# (ou com av-aloha/data_collection_scripts no PYTHONPATH), e reaproveita:
#   - webrtc_headset.WebRTCHeadset      (sinalização + vídeo + data channel)
#   - headset_control.HeadsetControl    (tradução de pose -> ação do braço)
#   - sim_env.make_sim_env              (ambiente MuJoCo)
#
# Diferença em relação a data_collection_scripts/record_sim_episodes.py:
# este script NÃO grava episódios em HDF5 e roda em loop contínuo, sem
# limite de passos — é feito para demonstração ao vivo, não coleta de dados.

raise NotImplementedError(
    "run_demo.py ainda não foi implementado. Ver README.md, seção 'Setup', passo 5."
)
