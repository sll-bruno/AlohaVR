#!/bin/bash
# Lançador da demo para os monitores da UPA: dois cliques neste arquivo.
#
# Reinicia sozinho se o processo cair, para que uma falha às 14h não exija
# alguém que saiba usar o terminal. Para encerrar: feche esta janela ou Ctrl+C.

cd "$(dirname "$0")" || exit 1
ALOHAVR="$PWD"
SIM="$(cd .. && pwd)/av-aloha/data_collection_scripts"
PY="$(cd .. && pwd)/av-aloha/venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "ERRO: ambiente Python nao encontrado em $PY"
  echo "Veja o README (secao Setup). Chame o responsavel tecnico."
  read -r -p "Pressione Enter para fechar."
  exit 1
fi
if [ ! -f "$SIM/serviceAccountKey.json" ]; then
  echo "ERRO: falta serviceAccountKey.json em $SIM"
  read -r -p "Pressione Enter para fechar."
  exit 1
fi

clear
cat <<'BANNER'
==========================================================
  AlohaVR - demo de teleoperacao VR
==========================================================
  1. Espere aparecer "Aguardando o oculos conectar"
  2. No oculos: abra o app, toque em Load, escolha o robo
     e toque em Connect
  3. Botao A: assumir o controle   |   Botao B: reiniciar
==========================================================
BANNER

while true; do
  cd "$SIM" || exit 1
  PYTHONPATH="$SIM" "$PY" -u "$ALOHAVR/run_demo.py" "$@"
  code=$?
  if [ $code -eq 0 ] || [ $code -eq 130 ]; then echo "Demo encerrada."; break; fi
  echo ""
  echo "A demo caiu (codigo $code). Reiniciando em 5 segundos..."
  echo "Se isso se repetir, chame o responsavel tecnico."
  sleep 5
done
