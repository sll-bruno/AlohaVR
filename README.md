# AlohaVR

**Vista o corpo de um robô. Sinta o que ele sente. Pilote com a sua cabeça e as suas mãos.**

AlohaVR é uma demonstração de teleoperação por VR de um robô bimanual simulado, construída em cima do projeto de pesquisa [AV-ALOHA](https://soltanilara.github.io/av-aloha/), da Soltani Lab. A ideia é simples: um operador veste um headset de VR, e passa a enxergar o mundo pelos "olhos" do robô — mexendo a cabeça, muda o ponto de vista; mexendo os braços, controla os braços do robô. Tudo isso rodando ao vivo sobre um robô simulado em física (MuJoCo), sem gravação nem treinamento de modelo envolvido — é só demonstração.

Este projeto foi apresentado na **UPA 2026**.

## O que este repositório contém

Este repositório reúne apenas as **adições feitas em cima dos projetos originais** — o loop de teleoperação em modo demo (sem gravação de dataset), instruções de setup e a definição de arquitetura da demo. Ele **não republica** o código do AV-ALOHA nem do app Unity: ambos são usados como dependências externas, clonados à parte.

## Créditos e dependências

Este projeto não existiria sem o trabalho original da Soltani Lab (UC Davis):

- **[AV-ALOHA](https://github.com/Soltanilara/av-aloha)** — código de simulação (MuJoCo), controle de teleoperação e treinamento de políticas (ACT). É de onde vem toda a física, o robô e a lógica de tradução de pose → ação de braço.
- **[av-aloha-unity](https://github.com/Soltanilara/av-aloha-unity)** — o app Unity que roda no headset (Meta Quest 2/3), responsável por capturar pose de cabeça/controles e exibir o vídeo estéreo recebido. É usado aqui sem modificação.
- **Paper**: ["Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation"](https://arxiv.org/abs/2409.17435)

Ambos os projetos originais têm sua própria licença — este repositório não reivindica autoria sobre o código deles, apenas sobre o que está listado abaixo.

## Arquitetura

```
[Pessoa com Quest 2]
   pose cabeça+mãos ──(WebRTC data channel)──► [run_demo.py, nesta máquina]
                                                        │
                                          headset_control.run(pose, estado_atual_braços)
                                                        │
                                                   env.step(action)  [MuJoCo, via av-aloha]
                                                        │
                                    ┌───────────────────┴───────────────────┐
                          render câmera de cabeça (estéreo)         render câmera 3ª pessoa
                                    │                                       │
                          (WebRTC, vídeo estéreo)                     janela local
                                    ▼                                       ▼
                         [Quest 2 — visão nos dois olhos]        [Tela/projetor da plateia]
```

O Quest roda apenas o app Unity — captura de pose e exibição de vídeo, nada de física ou lógica de robô. Toda a simulação, o controle e o servidor WebRTC rodam na máquina do operador.

A sinalização inicial (troca do SDP de offer/answer) usa Firestore, como no projeto original — vídeo e dados de controle trafegam depois disso direto por WebRTC (P2P), sem passar pelo Firebase.

## Setup

### 1. Clonar as dependências

Ao lado deste repositório, clone os dois projetos-base:

```bash
git clone https://github.com/Soltanilara/av-aloha.git
git clone https://github.com/Soltanilara/av-aloha-unity.git
```

### 2. Ambiente Python (av-aloha)

```bash
cd av-aloha
python3 -m venv venv && source venv/bin/activate
# instala as dependências do av-aloha (mujoco, gymnasium, aiortc, google-cloud-firestore, opencv-python etc.)
```

### 3. Firebase (sinalização)

- Cria um projeto no [Firebase Console](https://console.firebase.google.com), ativa o Firestore.
- Gera uma chave de conta de serviço e salva como `serviceAccountKey.json`.
- Cria `signalingSettings.json`:
  ```json
  { "robotID": "robot1", "password": "calls", "turn_server_url": "", "turn_server_username": "", "turn_server_password": "" }
  ```
- Copia os dois arquivos para `av-aloha/data_collection_scripts/`.

**Nunca commite `serviceAccountKey.json` nem `signalingSettings.json`** — o `.gitignore` deste repo já os ignora.

### 4. APK no Quest 2

```bash
cd av-aloha-unity
git lfs install && git lfs pull
adb install -r TwoStreamGuidedVision.apk
```

### 5. Rodar a demo

```bash
python run_demo.py
```

No headset: preenche Project ID / Password, aperta **Load**, escolhe o robô, aperta **Connect**, e segura o botão de engatar controle.

## Status

🚧 Em desenvolvimento para a apresentação na UPA 2026.

## Licença

O código deste repositório (as adições listadas acima) está sob licença MIT — veja [LICENSE](LICENSE). O uso do código do AV-ALOHA e do av-aloha-unity segue as licenças dos respectivos projetos originais.
