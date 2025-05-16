import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pickle
import os
import sys

# Inicialização dos pesos --------------------
if os.path.exists("pesos_dino.pkl"):
    with open("pesos_dino.pkl", "rb") as f:
        z = pickle.load(f)
    print("Pesos carregados do arquivo.")
else:
    z = [random.uniform(-1, 1) for _ in range(6)]
    print("Pesos iniciados aleatoriamente.")

learning_rate_passou = 1e-4 
learning_rate_colidiu = 1e-4

# Funções -------------------------------------
def tangente_hiperbolica(x):
    return math.tanh(x)

def derivada_tanh(u):
    # d/du tanh(u) = 1 - tanh(u)^2
    return 1 - math.tanh(u)**2


def salvar_pesos():
    global z
    
    filename = "pesos_dino.pkl"
    # Cria diretório se não existir (caso você queira salvar em subpastas)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(z, f)
    print(f"Pesos salvos com sucesso em '{filename}'")

def rede(distancia, velocidade_obs, altura_cacto, energia, largura_cacto, bias, w0, w1, w2, w3, w4):
      
    w = [w0, w1, w2, w3, w4]   
    soma = bias + w[0]*distancia + w[1]*velocidade_obs + w[2]*altura_cacto + w[3]*energia + w[4]*largura_cacto

    return soma

def verifica(z):
    return 1 if z > 0.2 else 0

def atualiza(z, entrada, lr, esperado):
    bias, w0, w1, w2, w3, w4 = z
    x0, x1, x2, x3, x4       = entrada  # ← espera 5 valores

    u = bias + w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4
    y = math.tanh(u)

    erro  = y - esperado
    delta = erro * (1 - math.tanh(u)**2)

    bias -= lr * delta
    w0   -= lr * delta * x0
    w1   -= lr * delta * x1
    w2   -= lr * delta * x2
    w3   -= lr * delta * x3
    w4   -= lr * delta * x4

    return [bias, w0, w1, w2, w3, w4]

# Plotar o gráfico de convergência
def grafico_convergencia(cost_values, iterations, soma): 
    '''
    # Valor atual da função de custo
    current_cost = soma  # Substitua pelo valor real da função de custo atual
    # Plotar o gráfico de convergência
    plt.plot(iterations, cost_values)
    plt.scatter(iterations[-1], current_cost, color='red', label='Valor Atual')
    '''    
    plt.plot(iterations, cost_values)
    plt.xlabel('Número de Iterações')
    plt.ylabel('Função de Custo')
    plt.title('Convergência da IA')
    plt.show()

#-----------------------------------------
# Inicialização do Pygame
pygame.init()


# Configurações / Variáveis --------------
largura_tela = 1500
altura_tela  = 500
tela = pygame.display.set_mode((largura_tela, altura_tela), pygame.HWSURFACE)
pygame.display.set_caption("Jogo do Dinossauro")
relogio = pygame.time.Clock()
FPS = 120

# — Cores —
BRANCO = (255, 255, 255)
PRETO  = (0, 0, 0)

# — Dinossauro —
dino_pos_x      = 50
dino_pos_y      = altura_tela - 50
dino_vel_y      = 0
potencia_pulo   = -3
dino_gravidade  = 0.05
dino_tamanho    = 50
dino_img        = pygame.image.load("dino.png")
dino_img        = pygame.transform.scale(dino_img, (dino_tamanho, dino_tamanho))
energia         = 1.0

# — Obstáculo (e máximos para normalização) —
obstaculo_pos_x         = largura_tela
obstaculo_pos_y         = altura_tela - 50
largura_min_obstaculo   = 300
largura_max_obstaculo   = 450
obstaculo_altura_min    = 40
obstaculo_altura_max    = 60
obstaculo_vel_x_min     = 8
obstaculo_vel_x_max     = 16

# sorteio inicial
obstaculo_largura = random.randrange(largura_min_obstaculo, largura_max_obstaculo)
obstaculo_altura  = random.randrange(obstaculo_altura_min,    obstaculo_altura_max)
obstaculo_vel_x   = random.randrange(obstaculo_vel_x_min,     obstaculo_vel_x_max)
cacto_img         = pygame.image.load("cacto.png")
cacto_img         = pygame.transform.scale(cacto_img, (50, obstaculo_altura))

# — Estado do jogo —
game_over = False
pontuacao  = 0
distancia  = 0
perdeu     = 0

# — Rede neural: inicialização dos pesos —
bias0 = 0.0
w00   = 0.0
w10   = 0.0
w20   = 0.0
w30   = 0.0
w40   = 0.0
z     = [bias0, w00, w10, w20, w30, w40]

# learning rates
learning_rate_passou  = 0.001
learning_rate_colidiu = 0.001

# — Listas de erro e iterações (para gráficos) —
erro_passou      = []    # <— re-adicionada
erro_colidiu     = []    # <— re-adicionada
iterations_passou  = []
iterations_colidiu = []

# — Métricas temporárias de pulo (usadas pela IA) —
distancia_pulo  = 0.0
velocidade_pulo = 0.0

# — Contadores de número de updates —
numero_att_passou   = 0
numero_att_colidiu  = 0

# — Inicialização de métricas de acurácia —  
acuracia    = 0.0     # ← define antes de usar

#----------------------------------------------------------------------------------------
paused = False

# Loop principal do jogo
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

        elif event.type == pygame.KEYDOWN:
            # PAUSA / DESPAUSA com P
            if event.key == pygame.K_p:
                paused = True
                while paused:
                    for ev in pygame.event.get():
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                            paused = False
                        elif ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    relogio.tick(FPS)

            # SPACE => termina o jogo, salva pesos e gera gráficos
            elif event.key == pygame.K_SPACE:
                salvar_pesos()
                grafico_convergencia(erro_passou, iterations_passou, soma)
                grafico_convergencia(erro_colidiu, iterations_colidiu, soma)
                game_over = True

    #Distância
    distancia = obstaculo_pos_x - dino_pos_x  
                   
    # Atualizar posição do dinossauro
    dino_vel_y += dino_gravidade
    dino_pos_y += dino_vel_y
    if dino_pos_y > altura_tela - 50:
        dino_pos_y = altura_tela - 50
        dino_vel_y = 0  
                 
    # Atualizar posição do obstáculo
    obstaculo_pos_x -= obstaculo_vel_x    
    if obstaculo_pos_x < -obstaculo_largura:

        if acerto_consec == 0 and pontuacao > 0:        
            entrada = (
                distancia / largura_tela,
                obstaculo_vel_x / obstaculo_vel_x_max,
                obstaculo_altura / obstaculo_altura_max,
                energia,
                obstaculo_largura / largura_max_obstaculo
            )
            z = atualiza(z, entrada, learning_rate_colidiu, 0)
            #Dados para o grafico de convergencia
            numero_att_passou += 1
            erro_passou.append(z[6])
            iterations_passou.append(numero_att_passou)

        obstaculo_pos_x = largura_tela 
        obstaculo_largura = random.randrange(largura_min_obstaculo, largura_max_obstaculo)
        obstaculo_altura = random.randrange(obstaculo_altura_min, obstaculo_altura_max)
        obstaculo_vel_x = random.randrange(obstaculo_vel_x_min, obstaculo_vel_x_max)   

        acerto_consec += 1    
        pontuacao += 1 
        energia = 1        

    # normalização simples
    dist_norm   =  (distancia)          / largura_tela         # [0,1]
    vel_norm    =  obstaculo_vel_x      / obstaculo_vel_x_max  # [0,1]
    alt_norm    =  obstaculo_altura     / obstaculo_altura_max # [0,1]
    eng_norm    =  energia               # já em [0,1]
    larg_norm   =  obstaculo_largura    / largura_max_obstaculo# [0,1]

    # e depois use esses na rede:
    soma = rede(dist_norm, vel_norm, alt_norm, eng_norm, larg_norm,
                z[0], z[1], z[2], z[3], z[4], z[5])


    #Usando funções para rede
    soma = rede(distancia, obstaculo_vel_x, obstaculo_largura, energia, perdeu, z[0], z[1], z[2], z[3], z[4], z[5]) / 1000
    soma_original = soma
    soma = tangente_hiperbolica(soma) 
    x = verifica(soma)
    
    
    #Pulo automático
    if x == 1 and dino_pos_y == altura_tela - 50 and energia >= 1 and distancia > 100:        
        dino_vel_y = potencia_pulo     
        energia -= 1
        distancia_pulo = distancia
        velocidade_pulo = obstaculo_vel_x        
    

    # Colisão    
    # Atualizando pontos
    dino_inf_esq = [(dino_pos_x) , (dino_pos_y + 50)]
    dino_inf_dir = [(dino_pos_x + 50) , (dino_pos_y + 50)]
    dino_sup_esq = [(dino_pos_x) , (dino_pos_y)]
    dino_sup_dir = [(dino_pos_x + 50) , (dino_pos_y)]
    #------------------
    obstaculo_sup_esq = [obstaculo_pos_x, altura_tela-obstaculo_altura]
    obstaculo_sup_dir = [obstaculo_pos_x+(obstaculo_largura),altura_tela-obstaculo_altura]
    #------------------------------------------------------------------------


    #Verifica se tocou no chão depois de pular
    if (energia == 0) and (dino_inf_dir[1] == 500):
        toque_depois_pulo = 1
    else:
        toque_depois_pulo = 0

    if ((dino_inf_dir[0] >= obstaculo_sup_esq[0]) and (dino_inf_dir[0] <= obstaculo_sup_dir[0]) and (dino_inf_dir[1] >= obstaculo_sup_dir[1])) or ((dino_inf_esq[0] >= obstaculo_sup_esq[0]) and (dino_inf_esq[0] <= obstaculo_sup_dir[0]) and (dino_inf_esq[1] >= obstaculo_sup_esq[1])): 
        pygame.time.wait(1000) 
        if ((dino_inf_dir[0] >= obstaculo_sup_esq[0]) and (dino_inf_dir[0] <= obstaculo_sup_dir[0]) and (dino_inf_dir[1] >= obstaculo_sup_dir[1])) and ((dino_sup_dir[0] < obstaculo_sup_esq[0]) and (dino_sup_dir[0] < obstaculo_sup_dir[0]) and (dino_sup_dir[1] < obstaculo_sup_dir[1])):
            #Atualiza para esperar mais um pouco antes de pular
            entrada = (
                distancia / largura_tela,
                obstaculo_vel_x / obstaculo_vel_x_max,
                obstaculo_altura / obstaculo_altura_max,
                energia,
                obstaculo_largura / largura_max_obstaculo
            )
            z = atualiza(z, entrada, learning_rate_colidiu, 0)
            #Dados para o grafico de convergencia
            numero_att_passou += 1
            erro_passou.append(z[6])
            iterations_passou.append(numero_att_passou) 

        else:
            #Atualiza para esperar mais um pouco antes de pular
            entrada = (
                distancia / largura_tela,
                obstaculo_vel_x / obstaculo_vel_x_max,
                obstaculo_altura / obstaculo_altura_max,
                energia,
                obstaculo_largura / largura_max_obstaculo
            )
            z = atualiza(z, entrada, learning_rate_colidiu, 0)
            #Dados para o grafico de convergencia
            numero_att_colidiu += 1
            erro_colidiu.append(z[5])
            iterations_colidiu.append(numero_att_colidiu) 

        obstaculo_largura = random.randrange(largura_min_obstaculo, largura_max_obstaculo)
        obstaculo_altura = random.randrange(obstaculo_altura_min, obstaculo_altura_max)
        obstaculo_vel_x = random.randrange(obstaculo_vel_x_min, obstaculo_vel_x_max)

        acerto_consec = 0
        dino_pos_y = altura_tela - 50
        obstaculo_pos_x = largura_tela + 50        
        perdeu += 1
        energia = 1  

    #------------------------------------------------------------------------ Visuzalicação
    #Atualizando imagem do obstáculo
    cacto_img = pygame.transform.scale(cacto_img, (obstaculo_largura, obstaculo_altura))

    # Atualizando pontos
    dino_inf_esq = [(dino_pos_x) , (dino_pos_y + 50)]
    dino_inf_dir = [(dino_pos_x + 50) , (dino_pos_y + 50)]
    dino_sup_esq = [(dino_pos_x) , (dino_pos_y)]
    dino_sup_dir = [(dino_pos_x + 50) , (dino_pos_y)]
    #------------------
    obstaculo_sup_esq = [obstaculo_pos_x, altura_tela-obstaculo_altura]
    obstaculo_sup_dir = [obstaculo_pos_x+(obstaculo_largura), altura_tela-obstaculo_altura]
    #------------------------------------------------------------------------

    #Acurácia
    if perdeu > 0:
        acuracia = (pontuacao / (perdeu+pontuacao))
#----------------------------------------------------------------------------------------
    # Renderizar elementos na tela
    tela.fill(BRANCO)
    tela.blit(dino_img, (dino_pos_x, dino_pos_y))
    tela.blit(cacto_img, (obstaculo_pos_x, altura_tela-obstaculo_altura))      

    font = pygame.font.Font(None, 36)
    text = font.render("Soma ---> " + str(soma), True, PRETO)
    tela.blit(text, (10,  30))    

    font = pygame.font.Font(None, 36)
    text = font.render("Geração---> " + str(perdeu), True, PRETO)
    tela.blit(text, (10,  70)) 

    font = pygame.font.Font(None, 36)
    text = font.render("Distancia---> " + str(distancia), True, PRETO)
    tela.blit(text, (10, 90))  

    font = pygame.font.Font(None, 36)
    text = font.render("Vel cacto---> " + str(obstaculo_vel_x), True, PRETO)
    tela.blit(text, (10,  130)) 

    font = pygame.font.Font(None, 36)
    text = font.render("Altura cacto---> " + str(obstaculo_altura), True, PRETO)
    tela.blit(text, (10, 150)) 

    font = pygame.font.Font(None, 36)
    text = font.render("Largura cacto---> " + str(obstaculo_largura), True, PRETO)
    tela.blit(text, (10, 170)) 

    font = pygame.font.Font(None, 36)
    text = font.render("Soma Original ---> " + str(soma_original), True, PRETO)
    tela.blit(text, (10,  190)) 


    font = pygame.font.Font(None, 36)
    text = font.render("w0---> " + str(z[1]), True, PRETO)
    tela.blit(text, (400,  30)) 

    font = pygame.font.Font(None, 36)
    text = font.render("w1---> " + str(z[2]), True, PRETO)
    tela.blit(text, (400,  50)) 

    font = pygame.font.Font(None, 36)
    text = font.render("w2---> " + str(z[3]), True, PRETO)
    tela.blit(text, (400,  70)) 

    font = pygame.font.Font(None, 36)
    text = font.render("w3---> " + str(z[4]), True, PRETO)
    tela.blit(text, (400,  90)) 

    font = pygame.font.Font(None, 36)
    text = font.render("w4---> " + str(z[5]), True, PRETO)
    tela.blit(text, (400,  110)) 
    
    font = pygame.font.Font(None, 36)
    text = font.render("Bias---> " + str(z[0]), True, PRETO)
    tela.blit(text, (400,  130))     


    font = pygame.font.Font(None, 36)
    text = font.render("Pontos---> " + str(pontuacao), True, PRETO)
    tela.blit(text, (900,  30))

    font = pygame.font.Font(None, 36)
    text = font.render("Energia atual---> " + str(energia), True, PRETO)
    tela.blit(text, (900, 50))
    
    font = pygame.font.Font(None, 36)
    text = font.render("Acurácia atual ---> " + str(acuracia), True, PRETO)
    tela.blit(text, (900, 80))   

    #--------------------------------------------
    pygame.draw.circle(tela, (0, 0, 255), dino_sup_esq, 3)
    pygame.draw.circle(tela, (0, 0, 255), dino_sup_dir, 3)
    pygame.draw.circle(tela, (0, 0, 255), dino_inf_esq, 3)
    pygame.draw.circle(tela, (0, 0, 255), dino_inf_dir, 3)    
    #-----------------------   
    pygame.draw.circle(tela, (255,0,0), obstaculo_sup_dir, 3)
    pygame.draw.circle(tela, (255,0,0), obstaculo_sup_esq, 3) 
    #--------------------------------------------

    pygame.display.update()
# Encerrar o Pygame
pygame.quit()
