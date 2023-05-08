"""
Autor: Gabriel Vicente (20498)
Grupo: 5
Proyecto: Laboratorio 9 Q-learning en Frozen Lake

Descripción:
Implementar un agente inteligente que resuelva el juego de Frozen Lake
con el argumento de slippery=True, usando un algoritmo de aprendizaje por refuerzo.
"""

# Imports necesarios para el laboratorio 9
import gym
import numpy as np

# Titulo estetico

def banner(header, row_jumps = True):
    value   = 120
    banner = "{:─^120}".format(header)

    print(f'{"─"*value}')

    if row_jumps:
        print(f'\n{banner}\n')
    else:
        print(f'{banner}')
    print(f'{"─"*value}\n')

# Creacion del ambiente
FrozenLake = gym.make('FrozenLake-v1',is_slippery=True, render_mode="human")

# Variables necesarias para el Q-Learining
process                 = 50
sub_p                   = 25
learning_rate           = 0.9
discount_factor         = 0.9
medion                  = 1.0
x_m                     = 1.0
m_m                     = 0.1
discount                = 0.005

# Tabla de registro de Q-Table
n_states    = FrozenLake.observation_space.n
n_actions   = FrozenLake.action_space.n
final_q_t     = np.zeros((n_states, n_actions))

# Q-learning
for n in range(process):
    print("N analizado: ", n)
    estado_n = FrozenLake.reset()[0]
    estado_n_end = False
    for step in range(sub_p):
        if estado_n_end==True:
            break
        else:

            if np.random.uniform() > medion:
                doing = np.argmax(final_q_t[estado_n, :])
            elif np.random.uniform() <= medion:
                doing = FrozenLake.action_space.sample()
            follow_state, gain, estado_n_end, _, _ = FrozenLake.step(doing)
            follow_state = int(follow_state)
            final_q_t[estado_n, doing] = final_q_t[estado_n, doing] + learning_rate * (gain + discount_factor * np.max(final_q_t[follow_state, :]) - final_q_t[estado_n, doing])
            estado_n = follow_state

    medion = m_m + (x_m - m_m) * np.exp(-discount*n)
    
banner(" Q table")
print(final_q_t)

# Mejora progresiva del agente
n_test_episodes = 100
for n in range(n_test_episodes):
    estado_n = FrozenLake.reset()
    estado_n = estado_n[0]
    estado_n_end = False

    for step in range(sub_p):
        doing = np.argmax(final_q_t[estado_n, :])
        follow_state, gain, estado_n_end, _, _ = FrozenLake.step(doing)
        follow_state = int(follow_state)
        estado_n = follow_state
        if estado_n_end:
            break
