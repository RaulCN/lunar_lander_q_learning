import gymnasium as gym
import numpy as np
import pickle

# Função para carregar a tabela Q
def carregar_q_table(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Função para discretizar o estado contínuo
def discretizar_estado(state, bins):
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))

# Configurações do ambiente
env = gym.make('LunarLander-v2', render_mode='human')
q_table_filename = "q_table_lunarlander.pkl"

# Carrega a tabela Q
q_table = carregar_q_table(q_table_filename)

# Discretização dos estados
bins = [np.linspace(-1, 1, 20), np.linspace(-1.5, 1.5, 20), np.linspace(-1, 1, 20), 
        np.linspace(-1.5, 1.5, 20), np.linspace(-1, 1, 20), np.linspace(-1.5, 1.5, 20),
        np.linspace(-3.14, 3.14, 20), np.linspace(-1, 1, 20)]

# Loop principal para continuar tentando até pousar
while True:
    state, _ = env.reset()
    state = discretizar_estado(state, bins)

    for _ in range(1000):  # Limite de passos por episódio
        env.render()
        
        # Escolhe a melhor ação baseada na tabela Q
        action = np.argmax(q_table.get(state, np.zeros(env.action_space.n)))
        
        # Executa a ação no ambiente
        next_state, reward, done, info, _ = env.step(action)
        next_state = discretizar_estado(next_state, bins)
        
        state = next_state

        if done:  # Verifica se o episódio terminou (o agente pousou)
            if reward == 100:  # Recompensa máxima para pouso perfeito
                print("Pouso bem-sucedido!")
            else:
                print("O episódio terminou, mas o pouso não foi perfeito.")
            break  # Reinicia o ambiente para tentar novamente

# Fecha o ambiente ao final (se você decidir sair do loop)
env.close()
