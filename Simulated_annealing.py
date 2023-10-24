from numba import jit
import numpy as np
import time
import math

#define a distancia entre duas cidades quaisquer
@jit(nopython=True)
def distances(N,x,y):
    
    dist = np.zeros((N,N),dtype=np.float32)
    for i in range(N):
        for j in range(N):
            dist[i,j] = np.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))
            
    return dist

@jit(nopython=True)
def custo(N,path,dist):
    # calcula a distancia total percorrida pela caminhada
    ener = 0
    for i in range(N-1):
        ener += dist[path[i],path[i+1]]
    ener += dist[path[0],path[N-1]]     # conecta a última e a primeira cidades do caminho
    
    return ener

@jit(nopython=True)
def newpath(N,path):
    
    # define uma nova caminhada
    
    newpath = np.zeros(N,dtype=np.int16)

    i=np.random.randint(N)   # escolhe uma posição aleatória da caminhada
    j=i
    while j==i:
        j=np.random.randint(N)  # escolhe outra posição 
    if i>j:                    # ordena os índices
        ini = j
        fin = i
    else:
        ini = i
        fin = j

    for k in range(N):        # inverte o sentido em que percorre o caminho entre os indices escolhidos
        if k >= ini and k <= fin:
            newpath[k] = path[fin-k+ini]
        else:
            newpath[k] = path[k]

    return newpath,ini,fin

@jit(nopython=True)
def mcstep(N,beta,en,path,best_e,best_p,dist):
    # realiza um passo de Monte Carlo
    np1 = np.zeros(N,dtype=np.int16)
    
    np1,ini,fin = newpath(N,path) # propoe um novo caminho
    
    # determina a diferença de energia 
    esq = ini-1         # cidade anterior a inicial
    if esq < 0: esq=N-1      # condicao de contorno
    dir = fin +1        # cidade apos a final
    if dir > N-1: dir=0      # condicao de contorno
    de = -dist[path[esq],path[ini]] - dist[path[dir],path[fin]]+ dist[np1[esq],np1[ini]] + dist[np1[dir],np1[fin]]

    if de < 0:         # aplica o criterio de Metropolis
        en += de
        path = np1
        if en < best_e:  # guarda o melhor caminho gerado até o momento
            best_e = en
            best_p = path
    else:              # aplica o criterio de Metropolis
        if np.random.random() < np.exp(-beta*de):
            en += de
            path = np1
            
    return en,path,best_e,best_p

def manysteps(num_pontos, path, dist, temp_inicial=10, temp_final=0.0001, step=0.8):
    energys_arr=[]
    paths_arr=[]
    temp_arr=[]
    en=custo(num_pontos, path, dist)
    best_e=en
    best_p=path
    temp=temp_inicial
    while(temp>=temp_final):
        for _ in range(100):
            en, path, best_e, best_p=mcstep(num_pontos, 1/temp, en, path, best_e, best_p, dist)
            energys_arr.append(en)
            paths_arr.append(path)
        temp_arr.append(temp)
        temp=temp*step
    return energys_arr, paths_arr, best_e, best_p, temp_arr

def main():
    start_time = time.time()
    x=[]
    y=[]
    #leio arquivo
    with open("posicoes.dat", 'r') as arquivo:
        linhas = arquivo.readlines()
    for linha in linhas:
        num1,num2=map(float,linha.split())
        x.append(num1)
        y.append(num2)
    N=len(x)
    #crio caminho inicial aleatorio
    path_ini = np.zeros(N,dtype=np.int16)
    for i in range(N):
        path_ini[i]=i
    #Simulated Annealing
    dist=distances(N,x,y)
    energias, caminhos, melhor_energia, melhor_caminho, temp_arr = manysteps(N,path_ini, dist, temp_inicial=3.5, step=0.99, temp_final=0.0001)
    print(melhor_energia)
    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    main()
