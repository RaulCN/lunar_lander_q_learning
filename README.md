Projeto LunarLander-v2 com Q-Learning


Descrição:

Este projeto implementa um agente de aprendizado por reforço usando Q-Learning para resolver o ambiente LunarLander-v2 do Gymnasium (anteriormente OpenAI Gym). O objetivo é treinar um agente para pousar com sucesso um módulo lunar em uma plataforma designada.

Estrutura do Projeto
O projeto consiste em três arquivos principais:

LunarLander-v2_treinamento.py - Script para treinar o agente
LunarLander-v2_Inteligente.py - Script para visualizar o agente treinado em ação
q_table_lunarlander.pkl - Arquivo de pesos do modelo treinado (disponível via Google Drive)

Arquivo de Pesos
Devido ao tamanho do arquivo de pesos, ele está hospedado externamente:
Download q_table_lunarlander.pkl https://drive.google.com/file/d/1zhUcjNjyPrnOJcoifCD1eVebd0GcpZJm/view?usp=sharing

Requisitos:
gymnasium
numpy
pickle

Como Usar
1. Configuração

Clone o repositório
Instale as dependências:
pip install gymnasium numpy

Baixe o arquivo de pesos do Google Drive e coloque-o na mesma pasta dos scripts

2. Executar o Agente Treinado
Copypython LunarLander-v2_Inteligente.py
3. Treinar um Novo Agente (opcional)
python LunarLander-v2_treinamento.py

Detalhes Técnicos:
Processo de Treinamento

Episódios: 100,000
Passos máximos por episódio: 1,000
Taxa de aprendizado: 0.1
Fator de desconto: 0.99
Exploração: Decaiimento de 1.0 a 0.01 com fator de 0.995

Obs: Esse foi o último algorítimo utilizado, utilizei vários outros que não estão upados no projeto, então não tenho absoluta certeza da correlação do treinamento com a calibragem dos pesos na saída do treinamento.

Discretização do Estado
O ambiente contínuo é discretizado em 20 bins para cada uma das 8 dimensões do estado:

Posição horizontal (-1 a 1)
Posição vertical (-1.5 a 1.5)
Velocidade horizontal (-1 a 1)
Velocidade vertical (-1.5 a 1.5)
Ângulo (-1 a 1)
Velocidade angular (-1.5 a 1.5)
Contato da perna esquerda (-3.14 a 3.14)
Contato da perna direita (-1 a 1)

Como Funciona:

Treinamento
O agente utiliza Q-Learning, um algoritmo de aprendizado por reforço que cria uma tabela (Q-table) para aprender quais ações são melhores em cada estado. Como o LunarLander tem um espaço de estados contínuo, implementamos a discretização desse espaço em 20 bins para cada dimensão, tornando possível usar a tabela Q. Durante o treinamento, a política ε-greedy é usada para balancear exploração (tentar novas ações) e aproveitamento (usar o conhecimento já adquirido): inicialmente, o agente explora mais (ε = 1.0) e gradualmente passa a aproveitar mais o conhecimento adquirido (ε decai até 0.01). A tabela Q é atualizada constantemente usando a fórmula de Bellman, que considera a recompensa imediata e futuras recompensas potenciais. O ambiente fornece recompensas positivas para pousos bem-sucedidos e negativas para colisões entre outros movimentos indesejáveis, permitindo que o agente aprenda progressivamente a melhor estratégia para pousar o módulo lunar com segurança.

Execução:

O agente carrega a tabela Q pré-treinada q_table_lunarlander.pkl
Escolhe ações baseadas nos valores máximos da tabela Q
Demonstra a capacidade de pousar o módulo lunar



Resultados:
O agente treinado é capaz de:

Controlar o módulo lunar de forma eficiente
Realizar pousos suaves na plataforma designada
Adaptar-se a diferentes condições iniciais

Limitações e Possíveis Melhorias:

A discretização do espaço de estados, embora necessária para a implementação da tabela Q, pode limitar a precisão das ações do agente. Técnicas mais avançadas como Deep Q-Learning poderiam ser implementadas para lidar diretamente com o espaço contínuo, potencialmente alcançando melhores resultados. O tempo de treinamento é significativo devido à natureza da tabela Q e à necessidade de explorar um grande espaço de estados.
Existem várias possibilidades de otimização e experimentação:

Ajuste de Parâmetros:

Modificar o número de bins na discretização
Experimentar diferentes taxas de aprendizado e fatores de desconto
Ajustar a taxa de decaimento da exploração


Aprendizado Curricular e Adaptativo:

Implementar um sistema de dificuldade progressiva
Desenvolver algoritmos que se adaptam durante o treinamento
Usar diferentes abordagens baseadas no desempenho atual


Otimização da Q-table:

Implementar técnicas de limpeza para remover dados irrelevantes
Comprimir a Q-table mantendo apenas informações essenciais
Criar métodos para identificar e remover estados redundantes


Transferência e Gestão de Conhecimento:

Utilizar Q-tables otimizadas como ponto de partida
Desenvolver sistemas de mesclagem inteligente de diferentes Q-tables
Implementar métodos de "esquecimento seletivo" de dados menos úteis


Objetivos Especializados:

Treinar agentes para diferentes objetivos específicos
Criar um sistema de meta-aprendizado que escolhe a melhor abordagem
Desenvolver métricas mais sofisticadas para avaliar o desempenho



Estas melhorias poderiam resultar em um sistema mais eficiente, com melhor desempenho e uso mais otimizado de recursos computacionais. A implementação de algoritmos mais inteligentes e adaptativos, combinada com técnicas de otimização de dados, pode levar a um agente mais capaz e eficiente.

Contribuindo
Sinta-se à vontade para fazer fork do projeto e submeter pull requests com melhorias.

Obs: esse experimento não tem uma metodologia de testes de desempenho e análise da evolução dos algorítimos, é apenas um exercício de implementação de conceitos de aprendizado por reforço.
