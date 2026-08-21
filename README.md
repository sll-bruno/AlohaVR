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
| **Segurar A** | Assume o controle |
| **Segurar X** (~1,5 s) | Reinicia a cena e as referências |
| **B** | Volta à tela inicial (nativo do app) |

Engatar re-referencia cada membro na pose em que ele está: o movimento é **relativo**, não absoluto, e cada mão tem sua própria âncora. Por isso **não importa onde suas mãos estão ao apertar A** — o robô continua de onde parou. Solte e aperte de novo para "recentrar" quando seu braço chegar ao limite físico.

O **B já é usado pelo próprio app** ([TransitionStartScene.cs:13](https://github.com/Soltanilara/av-aloha-unity/blob/main/Guided-Vision/Assets/Scripts/PassthroughScene/TransitionStartScene.cs)) para sair da cena de teleop, o que foi descoberto testando: um reinício mapeado nele nunca funcionava, porque o Unity trocava de cena no mesmo frame.

### O que aparece dentro do óculos

| Quando | O quê | Onde |
|---|---|---|
| Primeiros 10 s, ou até apertar A | Tabela de controles | Presa à tela |
| Permanente | Instrução da tarefa | **Ancorada no mundo**, ao fundo da bancada |
| Segurando X | Rodinha de progresso | **Ancorada**, logo acima da instrução |
| Ao concluir | "Parabéns! Você concluiu a tarefa" | Presa à tela, ~6 s |

O que permanece em vista é ancorado no mundo: uma legenda presa à tela enquanto a cena se move é o conflito que causa enjoo em VR. O que é momentâneo pode seguir a cabeça — e no caso do parabéns **precisa**, porque quem acabou de encaixar está olhando para a mesa e não veria uma mensagem fixa ao fundo.

## Desempenho

Medido num MacBook Air M-series. O gargalo **não é GPU**: render roda em OpenGL e custa ~11 ms por olho a 1280×720. O custo está no `mj_step`, que é single-thread.

Perfil do passo de física original — 3,25 dos 3,29 ms em **colisão narrow-phase**, contra 0,014 ms de solver. Duas otimizações atacam exatamente isso:

| Mudança | Efeito |
|---|---|
| `MULTICCD` desligado | 342 → 94 contatos, física 2,85x mais rápida |
| Malhas de colisão do `world` → caixas | 24,3 → 19,1 ms (−21%) |
| Elos dos braços → caixas (`--collision all`) | −35% no total, mas pode gerar colisão falsa |

O modelo mantém geoms de visual e colisão **separados**, então trocar colisão por caixas não muda nada na tela. Os **dedos das garras preservam malha exata em todos os modos**.

Uma terceira otimização veio depois: **em repouso os olhos não são renderizados** (~25 ms/frame). Ninguém está olhando, e num Mac sem ventoinha ao longo de 8 h esse calor degrada os grupos seguintes.

Resultado: **9 Hz a 0,38x do tempo real → 20 Hz em tempo real**.

### Ajustes disponíveis

| Flag | Padrão | Efeito |
|---|---|---|
| `--collision` | `world` | `mesh` (original), `world`, `all` |
| `--physics-timestep` | `0.0025` | Maior = mais rápido, diverge sob contato acima de ~0.005 |
| `--substeps` | 20 | Menos substeps com timestep maior = mais rápido, menos fiel |
| `--eye-width/height` | 1280×720 | Menor alivia render **e** encode VP8 |
| `--fovy` | 70 | Campo de visão dos olhos (52 é a geometria exata do painel; 70 foi o validado no headset) |
| `--multiccd` | off | Religa contatos múltiplos: agarre mais firme, 2,8x mais lento |
| `--hole-margin` | 3 mm | Folga do encaixe (3 mm dá 1,1 cm por lado; 0 mantém o original) |
| `--socket-mass` | 160 g | Massa do alvo do encaixe — leve demais ele foge ao ser tocado |
| `--success-reset` | 15 s | Tempo até reiniciar depois de concluir |
| `--motion-scale` | 1.0 | Amplia o deslocamento das mãos (a cabeça segue sempre 1:1) |
| `--idle-reset` | 25 s | Rede de segurança se a pose parar de chegar |

**Feche outros aplicativos.** Chrome e o WindowServer disputando CPU/GPU chegaram a dobrar o custo de cada frame nas medições — é a maior alavanca isolada, e é grátis.

Para avaliar outra máquina, rode `bench_physics.py` nela e compare. CPU single-thread é o que importa; GPU fraca não é impedimento.

## Limitações conhecidas

**Borda preta no headset.** O painel de vídeo da cena Unity ocupa 81,9° × 52° (1280×720 unidades × escala 0,00135 a 1 m), enquanto o Quest 2 enxerga ~90° × 93°. Sobram faixas pretas. Corrigir exige abrir o projeto no Unity 2022.3.20f1, aumentar o painel, ajustar `--fovy` junto e recompilar o APK.

**Ajustes de jogabilidade que se afastam do modelo original.** A demo altera a cena para caber em dois minutos: a folga do encaixe passa de 8 mm para 1,1 cm por lado, o alvo pesa 160 g em vez de 101 g, e a colisão do marcador `pin` é desligada — ele tem a mesma seção da peça e ficava atravessado no tubo, travando a inserção a 2 cm de profundidade (a detecção de sucesso passou a ser geométrica e não depende mais de contato). Todos são flags: `--hole-margin 0 --socket-mass 0` volta ao comportamento do av-aloha.

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
| Novo óculos conecta (voltou pra tela inicial e apertou Connect de novo) | Cena reinicia automaticamente |
| Segurar **X** (esquerdo) por ~1,5 s | Cena reinicia por completo, incluindo a referência de cabeça/mãos |
| Física diverge | Cena reinicia, processo continua |
| Conexão cai de verdade (app fecha, sai da cena, crash) | Offer é republicado; o próximo conecta sozinho |
| Pose para de chegar por >25 s (`--idle-reset`) | Cena reinicia — rede de segurança, não o caminho principal |

**Importante, testado ao vivo com o headset real:** só tirar o óculos da cabeça e recolocar **não reinicia nada**. O `WebRTCStreamer.cs` só derruba a conexão no `OnDestroy()` — ou seja, quando o app sai da cena de teleop de volta pra tela inicial — e não tem nenhum handler de `OnApplicationPause`. A tela apaga porque o próprio sistema do Quest apaga o display quando ninguém está com o headset no rosto, mas o app continua rodando e a pose dos controles continua chegando o tempo todo. **A troca limpa entre alunos é o `hold X`**, não tirar/recolocar o headset.

O caso de a conexão cair de verdade (item da tabela acima) também não é comportamento do projeto original: o `WebRTCHeadset` do av-aloha só se recupera de `iceConnectionState == "closed"`. Testado ao vivo: o app *dispara* isso sozinho ao sair da cena — mas se algum dia isso falhar (ex: queda de Wi-Fi sem fechamento limpo), o `ConnectionWatchdog` assume, verificando se o mecanismo original já resolveu (por identidade do objeto de conexão) antes de agir, pra não derrubar uma sessão que acabou de ficar boa.

**Tela do público:** legenda com a tarefa e um indicador de conexão (amarelo = aguardando óculos, verde = conectado), para o monitor saber o estado sem olhar o terminal.

**Dentro do óculos:** tabela de controles nos primeiros segundos, instrução da tarefa sempre visível, e comemoração ao concluir. Detalhes na seção de controles.

**Som:** a comemoração toca no Mac, não no headset — o app só aceita tracks de vídeo (`OnTrack` filtra `TrackKind.Video`), então levar áudio ao óculos exigiria recompilar o APK. Serve de sinal para a plateia e o monitor.

