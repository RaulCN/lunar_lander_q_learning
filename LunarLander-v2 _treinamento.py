import gymnasium as gym  # Importa a biblioteca gym para simulação de ambientes de aprendizado por reforço
import numpy as np  # Importa a biblioteca NumPy para operações numéricas
import os  # Importa a biblioteca os para manipulação de arquivos
import pickle  # Importa a biblioteca pickle para serialização de objetos

# Função para discretizar o estado contínuo
def discretizar_estado(state, bins):
    # Converte cada elemento do estado em um índice baseado nos bins
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))

# Função para carregar a tabela Q, se existir
def carregar_q_table(filename):
    # Verifica se o arquivo da tabela Q existe
    if os.path.exists(filename):
        # Se existir, carrega a tabela Q do arquivo
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return {}  # Retorna um dicionário vazio se não houver arquivo

# Função para salvar a tabela Q
def salvar_q_table(q_table, filename):
    # Salva a tabela Q em um arquivo usando pickle
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

# Configurações do ambiente e do Q-Learning
env = gym.make('LunarLander-v2')  # Cria o ambiente LunarLander-v2
num_episodes = 100000  # Número total de episódios de treinamento
max_steps = 1000  # Número máximo de passos por episódio
learning_rate = 0.1  # Taxa de aprendizado para atualização da tabela Q
discount_factor = 0.99  # Fator de desconto para recompensas futuras
exploration_rate = 1.0  # Taxa de exploração inicial
exploration_decay = 0.995  # Fator de decaimento da taxa de exploração
min_exploration_rate = 0.01  # Taxa mínima de exploração
q_table_filename = "q_table_lunarlander.pkl"  # Nome do arquivo para salvar a tabela Q

# Cria ou carrega a tabela Q
q_table = carregar_q_table(q_table_filename)

# Discretização dos estados
bins = [
    np.linspace(-1, 1, 20),    # Bins para posição horizontal
    np.linspace(-1.5, 1.5, 20),  # Bins para posição vertical
    np.linspace(-1, 1, 20),    # Bins para velocidade horizontal
    np.linspace(-1.5, 1.5, 20),  # Bins para velocidade vertical
    np.linspace(-1, 1, 20),    # Bins para ângulo
    np.linspace(-1.5, 1.5, 20),  # Bins para velocidade angular
    np.linspace(-3.14, 3.14, 20),  # Bins para contato com as pernas
    np.linspace(-1, 1, 20)      # Bins para outra dimensão de contato
]

# Treinamento do agente
for episode in range(num_episodes):
    state, _ = env.reset()  # Reseta o ambiente e obtém o estado inicial
    state = discretizar_estado(state, bins)  # Discretiza o estado
    total_reward = 0  # Inicializa a recompensa total do episódio

    for _ in range(max_steps):
        # Escolhe a ação usando a política ε-greedy
        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()  # Seleciona uma ação aleatória
        else:
            # Seleciona a melhor ação baseada na tabela Q
            action = np.argmax(q_table.get(state, np.zeros(env.action_space.n)))

        # Executa a ação no ambiente
        next_state, reward, done, info, _ = env.step(action)
        next_state = discretizar_estado(next_state, bins)  # Discretiza o próximo estado
        total_reward += reward  # Acumula a recompensa total

        # Atualiza a tabela Q
        old_value = q_table.get(state, np.zeros(env.action_space.n))[action]  # Valor antigo
        next_max = np.max(q_table.get(next_state, np.zeros(env.action_space.n)))  # Melhor valor futuro

        # Calcula o novo valor usando a fórmula de atualização da tabela Q
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table.setdefault(state, np.zeros(env.action_space.n))[action] = new_value  # Atualiza a tabela Q

        state = next_state  # Atualiza o estado atual

        if done:
            break  # Termina o episódio se a condição de término for atingida

    # Reduz a taxa de exploração ao longo do tempo
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    # Imprime o progresso do treinamento
    print(f"Episódio: {episode+1}/{num_episodes}, Recompensa Total: {total_reward}")

# Salva a tabela Q após o treinamento
salvar_q_table(q_table, q_table_filename)

# Fecha o ambiente
env.close()

print("Treinamento concluído e modelo salvo!")  # Mensagem de conclusão
