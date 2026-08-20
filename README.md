# AlohaVR

**Vista o corpo de um robô. Pilote com a sua cabeça e as suas mãos.**

AlohaVR é uma demonstração de teleoperação por VR de um robô bimanual simulado, construída sobre o projeto de pesquisa [AV-ALOHA](https://soltanilara.github.io/av-aloha/), da Soltani Lab. Um operador veste um Meta Quest e passa a enxergar pelos "olhos" do robô: mover a cabeça muda o ponto de vista, mover os braços move os braços do robô. Tudo ao vivo, sobre física simulada (MuJoCo), sem gravação de dataset nem modelo de IA no meio — é controle direto.

Apresentado na **UPA 2026**.

## O que este repositório contém

Apenas as **adições sobre os projetos originais**: o loop de teleop em modo demo, um benchmark de viabilidade de máquina, e a configuração de sinalização. Ele **não republica** o código do AV-ALOHA nem do app Unity — ambos são clonados à parte.

| Arquivo | Função |
|---|---|
| `run_demo.py` | Loop de teleop ao vivo: recebe pose do Quest, aplica no robô, devolve vídeo estéreo |
| `bench_physics.py` | Mede se uma máquina aguenta a demo, sem precisar de headset |
| `firebase/` | Regras do Firestore usadas na sinalização |

## Créditos

- **[AV-ALOHA](https://github.com/Soltanilara/av-aloha)** — simulação MuJoCo, controle de teleoperação, treino de políticas. Origem de toda a física e da matemática de pose → ação.
- **[av-aloha-unity](https://github.com/Soltanilara/av-aloha-unity)** — app Unity do headset. Usado **sem modificação**.
- **Paper**: ["Active Vision Might Be All You Need"](https://arxiv.org/abs/2409.17435)

## Arquitetura

```
[Quest]  --pose cabeça+mãos (WebRTC data channel)-->  [run_demo.py]
                                                            |
                                          HeadsetFullControl -> env.step()  [MuJoCo]
                                                            |
                                     +----------------------+----------------------+
                            2 câmeras estéreo                          câmera 3ª pessoa
                                     |                                            |
                          (WebRTC, VP8, 2 tracks)                          janela local
                                     v                                            v
                          [Quest, um olho cada]                      [tela do público]
```

O Quest roda **só o app Unity** — captura de pose e exibição de vídeo. Física, controle e servidor WebRTC rodam na máquina do operador. O Firestore só intermedeia a troca inicial de SDP; depois disso o tráfego é P2P.

## Setup

### 1. Dependências

```bash
git clone https://github.com/Soltanilara/av-aloha.git
git clone https://github.com/Soltanilara/av-aloha-unity.git
cd av-aloha && python3.11 -m venv venv && venv/bin/pip install -r requirements.txt
```

**Python 3.11+ é necessário**: em 3.9 o pacote `av` (dependência do `aiortc`) não tem wheel pronta e tenta compilar contra ffmpeg 7.

### 2. Firebase (sinalização)

Projeto Firestore em modo de teste, com regras abertas — o app Unity fala com a REST API **sem autenticação** ([WebRTCStreamer.cs:219](https://github.com/Soltanilara/av-aloha-unity/blob/main/Guided-Vision/Assets/Scripts/PassthroughScene/WebRTCStreamer.cs)), então não há como restringir sem alterar o app. As regras em `firebase/firestore.rules` têm expiração automática por isso.

Coloque em `av-aloha/data_collection_scripts/`:
- `serviceAccountKey.json` (chave de conta de serviço do Firebase)
- `signalingSettings.json`:
  ```json
  { "robotID": "robot1", "password": "calls",
    "turn_server_url": "stun:stun.l.google.com:19302",
    "turn_server_username": "", "turn_server_password": "" }
  ```

⚠️ `turn_server_url` **não pode ficar vazio** — o `aiortc` rejeita string vazia como URI malformada e o processo morre ao criar o data channel.

### 3. APK no Quest

O `TwoStreamGuidedVision.apk` está no repo Unity via Git LFS (`git lfs install && git lfs pull`). Instalar exige Developer Mode ativado **pela conta dona do headset**, no app Meta Horizon do celular — não há caminho alternativo: baixar o APK pelo navegador do Quest não adianta, porque o sistema não expõe instalador de pacotes ao usuário.

## Rodando

```bash
cd av-aloha/data_collection_scripts
PYTHONPATH=$PWD ../venv/bin/python -u /caminho/AlohaVR/run_demo.py
```

O `-u` importa: sem ele o stdout fica bufferizado e nada aparece. O `PYTHONPATH` também: o Python adiciona ao path a pasta do *script*, não a pasta atual.

Espere `WebRTC: Waiting for answer...`. **Só então** conecte pelo headset — reiniciar o script apaga o offer do Firestore e exige reconectar pelo app.

No Quest: Project ID e Password conforme seu `signalingSettings.json` → **Load** → escolha o robô → **Connect**.

Mac e Quest precisam estar **na mesma rede Wi-Fi**, sem isolamento de clientes. Sem TURN configurado, redes distintas provavelmente não fecham conexão.

### Controles

| Você | Robô |
|---|---|
| Cabeça | Braço do meio (câmera estéreo) |
| Controle esquerdo / direito | Braço esquerdo / direito |
| Gatilhos indicadores | Garras (analógico) |
| **Botão A** | Engata / desengata |

Engatar re-referencia as poses no instante do aperto: o movimento é **relativo**, não absoluto. Solte e aperte de novo para "recentrar" quando seu braço chegar ao limite físico.

O aviso *head out of sync* antes de engatar significa cabeça inclinada além de ~11° — nivele o olhar e ele some.

## Desempenho

Medido num MacBook Air M-series. O gargalo **não é GPU**: render roda em OpenGL e custa ~11 ms por olho a 1280×720. O custo está no `mj_step`, que é single-thread.

Perfil do passo de física original — 3,25 dos 3,29 ms em **colisão narrow-phase**, contra 0,014 ms de solver. Duas otimizações atacam exatamente isso:

| Mudança | Efeito |
|---|---|
| `MULTICCD` desligado | 342 → 94 contatos, física 2,85x mais rápida |
| Malhas de colisão do `world` → caixas | 24,3 → 19,1 ms (−21%) |
| Elos dos braços → caixas (`--collision all`) | −35% no total, mas pode gerar colisão falsa |

O modelo mantém geoms de visual e colisão **separados**, então trocar colisão por caixas não muda nada na tela. Os **dedos das garras preservam malha exata em todos os modos**.

Resultado: **9 Hz a 0,38x do tempo real → 16,3 Hz a 0,82x**.

### Ajustes disponíveis

| Flag | Padrão | Efeito |
|---|---|---|
| `--collision` | `world` | `mesh` (original), `world`, `all` |
| `--physics-timestep` | `0.0025` | Maior = mais rápido, diverge sob contato acima de ~0.005 |
| `--substeps` | 20 | Menos substeps com timestep maior = mais rápido, menos fiel |
| `--eye-width/height` | 1280×720 | Menor alivia render **e** encode VP8 |
| `--fovy` | 52 | Ângulo real que o painel do app ocupa |
| `--multiccd` | off | Religa contatos múltiplos: agarre mais firme, 2,8x mais lento |

**Feche outros aplicativos.** Chrome e o WindowServer disputando CPU/GPU chegaram a dobrar o custo de cada frame nas medições — é a maior alavanca isolada, e é grátis.

Para avaliar outra máquina, rode `bench_physics.py` nela e compare. CPU single-thread é o que importa; GPU fraca não é impedimento.

## Limitações conhecidas

**Borda preta no headset.** O painel de vídeo da cena Unity ocupa 81,9° × 52° (1280×720 unidades × escala 0,00135 a 1 m), enquanto o Quest 2 enxerga ~90° × 93°. Sobram faixas pretas. Corrigir exige abrir o projeto no Unity 2022.3.20f1, aumentar o painel, ajustar `--fovy` junto e recompilar o APK.

**Controle fino é difícil.** Consequência da latência residual: a imagem é presa à cabeça, então cada frame perdido vira arrasto. A correção estrutural seria reprojeção (*timewarp*) no app Unity, reprojetando o painel localmente conforme a pose entre frames recebidos — resolveria a sensação de latência sem precisar de mais FPS.

**Divergência da física.** Timesteps grandes divergem sob contato (`mjWARN_BADQACC`). O loop captura `PhysicsError` e reseta a cena em vez de derrubar o processo.

## Ideias não exploradas

- **Paralelizar física e render.** Hoje o loop é sequencial e o tempo é a soma; com render/encode em thread separada sobre uma cópia do estado, viraria o máximo — potencialmente ~2x.
- **Cápsulas em vez de caixas** nos elos dos braços: mesma economia sem o risco de colisão falsa.
- **MJX** (MuJoCo em JAX, GPU) não ajuda aqui: ganha em rodar milhares de simulações em paralelo, não em acelerar uma única instância interativa.

## Licença

MIT — veja [LICENSE](LICENSE). O uso do AV-ALOHA e do av-aloha-unity segue as licenças dos projetos originais.

## Operação no dia do evento

Pensado para grupos rotativos, monitores treinados numa única sessão e ninguém com acesso ao código por perto.

**Para iniciar:** dois cliques em `iniciar-demo.command`. Ele confere o ambiente, mostra as instruções na tela e **reinicia sozinho** se o processo cair.

**Entre um usuário e outro, nada precisa ser feito no computador:**

| Situação | O que acontece |
|---|---|
| Novo óculos conecta | Cena reinicia automaticamente |
| Alguém aperta **B** | Cena reinicia |
| Óculos sem uso por 25 s (`--idle-reset`) | Cena reinicia |
| Física diverge | Cena reinicia, processo continua |
| Óculos desconecta | Offer é republicado; o próximo conecta sozinho |

Esse último item é o mais importante e não é comportamento do projeto original: o `WebRTCHeadset` do av-aloha só se recupera de `iceConnectionState == "closed"`, estado que no aiortc só ocorre quando a conexão é fechada localmente. Quando alguém tira o óculos, o estado vai para `failed`, o offer nunca volta ao Firestore (que já foi apagado após a primeira conexão) e **o usuário seguinte não conseguiria conectar** sem reiniciar o processo. O `ConnectionWatchdog` cobre isso.

**Tela do público:** legenda com a tarefa e um indicador de conexão (amarelo = aguardando óculos, verde = conectado), para o monitor saber o estado sem olhar o terminal.

**Dentro do óculos:** a instrução da tarefa aparece o tempo todo, junto de "Segure A para começar".

