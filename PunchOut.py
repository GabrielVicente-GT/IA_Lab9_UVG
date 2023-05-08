"""
Autor: Gabriel Vicente (20498)
Grupo: 5
Proyecto: Laboratorio 9 Q-learning en PunchOut

Descripción:
Implementar un agente inteligente que resuelva el juego de Punch Out usando un algoritmo de aprendizaje por refuerzo.
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
PunchOut = gym.make('BoxingNoFrameskip-v4')

# Variables necesarias para el Q-Learining
n               = 0
process         = 100
sub_p           = 25
learning        = 0.9
discount_factor = 0.9
medion          = 1.0
x_m             = 1.0
m_m             = 0.1
discount        = 0.005


# Tabla de registro de Q-Table
n_states = gym.spaces.flatten_space(PunchOut.observation_space).shape[0]
n_actions = PunchOut.action_space.n
final_q_t = np.zeros((n_states, n_actions))

# Q-learning modified
while n < process:
    print("N analizado: ", n)
    estado_n = PunchOut.reset()
    estado_n = gym.spaces.flatten(gym.spaces.flatten_space(PunchOut.observation_space), estado_n[0])
    estado_n_end = False
    m_sub_p = 0
    while not estado_n_end and m_sub_p < sub_p:
        if np.random.uniform(0, 1) > medion:
            doing = np.argmax(final_q_t[estado_n, :])
        elif np.random.uniform(0, 1) <= medion:
            doing = PunchOut.action_space.sample()
        following, gain, estado_n_end, _, _ = PunchOut.step(doing)
        final_q_t[estado_n, doing] = final_q_t[estado_n, doing] + learning * (gain + discount_factor * np.max(final_q_t[following, :]) - final_q_t[estado_n, doing])
        estado_n = following
        m_sub_p += 1
    n += 1
    medion = m_m + (x_m - m_m) * np.exp(-discount * n)
PunchOut.close()

banner(" Q table")
print(final_q_t)

# Mejora progresiva del agente
estado_n_end = False
PunchOut = gym.make('BoxingNoFrameskip-v4',  render_mode="human")
estado_n = PunchOut.reset()
for _ in range(1000):
    if estado_n_end==True:
        break
    else:
        estado_n = gym.spaces.flatten(gym.spaces.flatten_space(PunchOut.observation_space), estado_n[0])
        doing = np.argmax(final_q_t[estado_n, :])
        following, gain, estado_n_end, _, _ = PunchOut.step(doing)
        estado_n = following
PunchOut.close()
