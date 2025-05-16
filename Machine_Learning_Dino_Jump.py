import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pickle
import os

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

def rede(distancia, velocidade_obs, altura_cacto, energia, largura_cacto, bias, w0, w1, w2, w3, w4):
      
    w = [w0, w1, w2, w3, w4]   
    soma = bias + w[0]*distancia + w[1]*velocidade_obs + w[2]*altura_cacto + w[3]*energia + w[4]*largura_cacto

    return soma

def verifica(z):
    return 1 if z > 0.2 else 0

def atualiza_passou(x, y, bias, w0, w1, w2, w3, w4, lr):
    distancia, vel, altura, energia, largura, esperado = x
    z = bias + w0 * distancia + w1 * vel + w2 * altura + w3 * energia + w4 * largura
    erro = z - esperado

    bias -= lr * erro * 1
    w0   -= lr * erro * distancia
    w1   -= lr * erro * vel
    w2   -= lr * erro * altura
    w3   -= lr * erro * energia
    w4   -= lr * erro * largura

    return [bias, w0, w1, w2, w3, w4, abs(erro)]


def atualiza_colidiu(dados, bias, w0, w1, w2, w3, w4, lr):
    distancia, vel, altura, energia, largura, esperado = dados
    z = bias + w0 * distancia + w1 * vel + w2 * altura + w3 * energia + w4 * largura
    erro = z - esperado

    bias += lr * erro
    w0   += lr * erro * distancia
    w1   += lr * erro * vel
    w2   += lr * erro * altura
    w3   += lr * erro * energia
    w4   += lr * erro * largura

    return [bias, w0, w1, w2, w3, w4, abs(erro)] 

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

# Configurações da tela
largura_tela = 1500
altura_tela = 500
tela = pygame.display.set_mode((largura_tela, altura_tela), pygame.HWSURFACE)
pygame.display.set_caption("Jogo do Dinossauro")

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)

# Variáveis do dinossauro
dino_pos_x = 50
dino_pos_y = altura_tela - 50
dino_vel_y = 0
potencia_pulo = -3   #-0.5
dino_gravidade = 0.05  #-0.001
dino_tamanho = 50
dino_img = pygame.image.load("dino.png")
dino_img = pygame.transform.scale(dino_img, (50, 50))
energia = 1

# Variáveis do obstáculo
obstaculo_pos_x = largura_tela
obstaculo_pos_y = altura_tela - 50
largura_max_obstaculo = 450
largura_min_obstaculo = 300
obstaculo_vel_x_min =8
obstaculo_vel_x_max = 16
obstaculo_altura_min = 40
obstaculo_altura_max = 60

obstaculo_largura = random.randrange(largura_min_obstaculo, largura_max_obstaculo)
obstaculo_altura = random.randrange(obstaculo_altura_min, obstaculo_altura_max)
obstaculo_vel_x = random.randrange(obstaculo_vel_x_min, obstaculo_vel_x_max)
cacto_img = pygame.image.load("cacto.png")
cacto_img = pygame.transform.scale(cacto_img, (50, obstaculo_altura))

# Estado do jogo
game_over = False
pontuacao = 0
distancia = 0
perdeu = 0

relogio = pygame.time.Clock()
FPS = 120  # Limitar a 60 quadros por segundo

#Data base
data_base_passou = []
data_base_colidiu = []
cont = 0
contador_velocidade = 0
soma = 0
acuracia= 0
acerto_consec = 0
perdeu_consec = 0
toque_depois_pulo = 0
distancia_pulo = 0
velocidade_pulo = 0

peso_bias = []
peso_w0 = []
peso_w1 = []
peso_w2 = []
peso_w3 = []
peso_w4 = []

#Para graficos
cost_values = [] 
iterations_colidiu = []
iterations_passou = []
erro_colidiu = []
erro_passou = []
numero_att_passou = 0
numero_att_colidiu = 0

#----------------------------------------------------------------------------------------
paused = False

# Loop principal do jogo
while not game_over: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:  # tecla P para pausar/despausar
                paused = not paused

            if event.key == pygame.K_SPACE and dino_pos_y == altura_tela - 50 and not paused:
                dino_vel_y = potencia_pulo
                energia -= 1
                distancia_pulo = distancia
                velocidade_pulo = obstaculo_vel_x 

            if event.key == pygame.K_g:
                grafico_convergencia(erro_passou, iterations_passou, soma)
                grafico_convergencia(erro_colidiu, iterations_colidiu, soma)

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
            entrada = (distancia_pulo, velocidade_pulo, obstaculo_altura, energia, obstaculo_largura, 1)
            z = atualiza_passou(entrada, z[0], z[1], z[2], z[3], z[4], z[5], learning_rate_passou)
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
        #data_base_passou.append((distancia_pulo, velocidade_pulo, obstaculo_altura, energia, obstaculo_largura, 1))   

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
            entrada = (distancia_pulo, velocidade_pulo, obstaculo_altura, energia, obstaculo_largura, 1)
            z = atualiza_passou(entrada, z[0], z[1], z[2], z[3], z[4], z[5], learning_rate_passou)
            #Dados para o grafico de convergencia
            numero_att_passou += 1
            erro_passou.append(z[6])
            iterations_passou.append(numero_att_passou) 

        else:
            #Atualiza para esperar mais um pouco antes de pular
            entrada = (distancia_pulo, velocidade_pulo, obstaculo_altura, energia, obstaculo_largura, 0)  # esperado = 0 para colisão
            z = atualiza_colidiu(entrada, z[0], z[1], z[2], z[3], z[4], z[5], learning_rate_colidiu)
            #Dados para o grafico de convergencia
            numero_att_colidiu += 1
            erro_colidiu.append(z[6])
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
