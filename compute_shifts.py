#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import functools
import tensorflow_probability as tfp
import random
from datetime import date
import time
import networkx as nx
import sys 

# Expected input:
# python compute_shifts.py MEASURE_ID [-ker 1 1 1 ] ELEMENTS
# python compute_shifts.py MEASURE_ID -ker 1 1 1  Ca K



# Set to 1 for debugging
FAST_TOTAL_LOSS = False
MAX_ITERATIONS = 300000
POS_TOL=1e-2
FUNC_TOL=1e-8



################ file handling ########################
    
PATH_TO_DATA="../newdata/"
def leading_zeros(number,length=4):
    """turns integer 42 into '0042'"""
    s=str(number)
    return '0'*(length-len(s))+s

def read_data(measure_id,element='Cu'):
    path=PATH_TO_DATA+leading_zeros(measure_id)+"/"
    imgs = []
    for i in range(100):  # meh
        try:
            filename = "{}_{}_{}.txt".format(leading_zeros(measure_id),leading_zeros(i+1),element)
            img = np.loadtxt(path+filename,delimiter=';',dtype='float32')
            imgs.append(img)
        except OSError:  # file not found
            continue
    return imgs

def construct_header(status):
    header=""
    for k,v in status.items():
        header+="{}: {}\n".format(k,v)
    return header

def construct_filename(status):
    mode=status.get("mode","unknown")
    mode=mode.split()[0]
    measure_id=leading_zeros(status.get("measure_id","xxxx"))
    element=status.get("element","Pu")
    today=date.today().strftime("%d.%m.%Y")
    rnd=leading_zeros(random.randint(0,9999))
    return "{}_shifts{}_{}_{}_{}.txt".format(mode,measure_id,element,today,rnd)

def save(shifts,status,folder='shifts/'):
    status['date']=date.today().strftime("%B %d, %Y")
    status['algorithm']='Nelder-Mead'
    status['function tolerance']=FUNC_TOL
    status['max iterations']=MAX_ITERATIONS    
    status['fast_total_loss']=FAST_TOTAL_LOSS
    filename = construct_filename(status)
    header = construct_header(status)
    np.savetxt(folder+filename,shifts,header = header)

############### registration ###########################

def running_time(f):
    def with_running_time(*args,**kwargs):
        global rtime
        start=time.time()
        result=f(*args,**kwargs)
        rtime = round(time.time()-start)
        #print("running time: {}s".format(rtime))
        return result
    return with_running_time

#@tf.function   ### non-eager execution fails, no .numpy method
def shift_loss_factory(img1,img2):
    def shifted_mse(shift):
        shifted_img=xrf.tfa_shift(img1,shift.numpy())
        return xrf.img_mse(img2,shifted_img,10)
    return shifted_mse

def shifted_total_mse(imgs,shifts):
    N=len(imgs)
    err=0
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            shift=shifts[2*i:2*i+2]-shifts[2*j:2*j+2]
            shifted_img=xrf.tfa_shift(imgs[j],shift.numpy())
            err+=xrf.img_mse(imgs[i],shifted_img,10)
    return err

def fast_shifted_total_mse(imgs,shifts):
    N=len(imgs)
    shifted_imgs = [xrf.tfa_shift(imgs[i],-shifts[2*i:2*i+2].numpy()) for i in range(N)]
    err=0
    for i in range(N):
        for j in range(i):
            err+=xrf.img_mse(shifted_imgs[i],shifted_imgs[j],10)
    return err


def total_shift_loss_factory(imgs):
    if FAST_TOTAL_LOSS:
        def loss(shifts):
            return fast_shifted_total_mse(imgs,shifts)
    else:
        def loss(shifts):
            return shifted_total_mse(imgs,shifts)
    return loss

def rnd_start(N=1):
    """generate 2N-dim random inital point close to the origin"""
    return tf.constant(np.random.random(2*N)-0.5,dtype='float32')

def zero_start(N=1):
    return tf.constant(np.zeros(2*N),dtype='float32')    

@running_time
def multiple_shift_with_nelder_mead(loss,start=rnd_start(),pos_tol=POS_TOL,tries=5):
    """run Nelder-Mead multiple times, each with different rnd start, return best result"""
    best=np.inf
    for _ in range(tries):
        result=tfp.optimizer.nelder_mead_minimize(
          loss,
          initial_vertex=start,
          step_sizes=1,
          func_tolerance=FUNC_TOL,
          position_tolerance=pos_tol,
          max_iterations=MAX_ITERATIONS)
        if result.objective_value<best:
            best_results=result
            best=result.objective_value
    return best_results

def center_of_mass_correction(shifts):
    center_of_mass=shifts.sum(axis=0)/len(shifts)
    return shifts-center_of_mass

def compute_pw_shifts(imgs,tries=5,pos_tol=POS_TOL,status={}):
    pw_shifts_nm={}
    N=len(imgs)
    feval = 0
    converged = True
    total_time = 0
    progbar = tf.keras.utils.Progbar(N)
    progbar.update(0)    
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            img,ref_img=imgs[i],imgs[j]
            result=multiple_shift_with_nelder_mead(shift_loss_factory(imgs[i],imgs[j]),start=rnd_start(),tries=tries,pos_tol=pos_tol)
            pw_shifts_nm[(i,j)]=result.position.numpy()
            feval += result.num_objective_evaluations.numpy()
            converged = converged and result.converged.numpy()
            total_time += rtime
        progbar.update(i+1)   
    status['total time']=total_time
    status['mode']="pw shifts"
    fill_status(status,tries,pos_tol,"rnd",feval,converged)
    positions_nm = np.array([1/N * sum(pw_shifts_nm[(j,i)] for j in range(N) if j != i) for i in range(N)])
    positions_nm=center_of_mass_correction(positions_nm)
    save(positions_nm,status)
    return positions_nm

def fill_status(status,tries,pos_tol,start,feval,converged):
    status['converged']=converged
    status['function evaluations']=feval
    status['tries']=tries
    status['position tolerance']=pos_tol
    status['start']=start
    
def compute_direct_registration_shifts(imgs,tries=1,pos_tol=POS_TOL,start="zero",status={}):
    N=len(imgs)
    if start=="zero":
        start_pos=zero_start(N=N)
    elif start=="rnd":
        start_pos=rnd_start(N=N)
    else:
        raise ValueError()
    loss=total_shift_loss_factory(imgs)
    result = multiple_shift_with_nelder_mead(loss,start=start_pos,tries=tries,pos_tol=pos_tol)
    direct_registration_shifts_nm = result.position.numpy()
    converged = result.converged.numpy()
    feval = result.num_objective_evaluations.numpy()
    fill_status(status,tries,pos_tol,start,feval,converged)
    status['total time']=rtime
    status['start']=start
    status['mode']="direct_registration"
    direct_registration_shifts_nm =center_of_mass_correction(direct_registration_shifts_nm.reshape(-1,2))
    save(direct_registration_shifts_nm,status)
    return direct_registration_shifts_nm
    
####### graph method ##################################
    
def compute_gap(G):
    spectrum=sorted(np.real(nx.adjacency_spectrum(G)))
    return spectrum[-1]/spectrum[-2]

def look_for_best_expander(n,d,tries=100,record=False):
    best=None
    best_gap=0
    records=[]
    for _ in range(tries):
        G=nx.random_regular_graph(d,n)
        gap=compute_gap(G)
        if gap>best_gap:
            best=G
            best_gap=gap
        if record:
            records.append(gap)
    if record:
        return best,best_gap,records
    return best,best_gap

def save_graph(G,filename,folder='shifts/',comments=[]):
    arr = nx.to_numpy_array(G)
    header = ""
    for comment in comments:
        header += comment + '\n'
    np.savetxt(folder+filename,arr,header = header)
    
def load_adj_of_graph(filename,folder='shifts/'):
    a =  np.loadtxt(folder+filename)
    return a

def find_and_save_graph(n,d,filename):
    best,best_gap = look_for_best_expander(n,d)
    comments = ['n = {}'.format(n),'d = {}'.format(d),'gap = {}'.format(best_gap)]
    save_graph(best,filename,comments=comments)

def shortest_paths(A):
    # route[i,j] = v bedeutet, dass der kürzeste ij Weg mit der Kante iv startet 
    # (und mit dem kürzesten vj Pfad fortgesetzt wird).
    N = len(A)
    dist = A.copy()+N*(1-A.copy())
    for i in range(N):
        dist[i,i]=0
    route = np.array([[j for j in range(N)]for i in range(N)])
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i,j]>dist[i,k]+dist[k,j]:
                    dist[i,j] = dist[i,k]+dist[k,j]
                    route[i,j] = route[i,k]
    return route
                    
def positions_from_graph(A,pw_shifts):
    N = len(A)
    route = shortest_paths(A)
    pos = np.zeros((N,2))
    for i in range(N):
        for j in range(N):
            k=j
            while k != i:
                pos[i] += pw_shifts[k,route[k,i]] # TODO: Pw shift berechnen & speichern
                k = route[k,i]
        pos[i] = pos[i]/N
    pos=center_of_mass_correction(pos)
    return pos

@running_time
def graph_shifts_helper(imgs,n,d,tries=5,pos_tol=POS_TOL,status={},A_fixed = None):
    filename = 'graph{}_{}_{}.txt'.format(n,d,time.strftime("%d.%m.%Y_%H%M%S")) # TODO: Sekunden einfügen
    if A_fixed is None:
        status['graph file']=filename
        find_and_save_graph(n,d,filename)
        A = load_adj_of_graph(filename,folder='shifts/')
    else:
        A = A_fixed
        
    # PW shifts bestimmen
    pw_shifts = np.array([[None for i in range(n)] for i in range(n)])
    feval = 0
    converged = True
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                img,ref_img=imgs[i],imgs[j]
                result=multiple_shift_with_nelder_mead(shift_loss_factory(imgs[i],imgs[j]),tries=tries,pos_tol=pos_tol)
                pw_shifts[i,j]=result.position.numpy()
                feval += result.num_objective_evaluations.numpy()
                converged = converged and result.converged.numpy()
    pos = positions_from_graph(A,pw_shifts)
    fill_status(status,tries,pos_tol,"rnd",feval,converged)    
    return pos

def compute_graph_shifts(imgs,d,tries=5,pos_tol=POS_TOL,status={}):
    N=len(imgs)
    graph_shifts=graph_shifts_helper(imgs,N,d,tries=tries,pos_tol=POS_TOL,status=status)
    status['total time']=rtime
    status['start']="rnd"
    status['mode']="graph shifts"
    status['graph degree']=d
    save(graph_shifts,status)
    return graph_shifts


def compute_path_shifts(imgs,tries=5,pos_tol=POS_TOL,status={}):
    N=len(imgs)
    d=1
    A_path = np.zeros((N,N))
    for i in range(N-1):
        A_path[i,i+1]=1
        A_path[i+1,i]=1
    graph_shifts=graph_shifts_helper(imgs,N,d,tries=tries,pos_tol=POS_TOL,status=status,A_fixed=A_path)
    status['total time']=rtime
    status['start']="rnd"
    status['mode']="Path shifts"
    save(graph_shifts,status)
    return graph_shifts


def compute_star_shifts(imgs,center,tries=5,pos_tol=POS_TOL,status={}):
    N=len(imgs)
    d=1
    A_star = np.zeros((N,N))
    for i in range(N):
        if i != center:
            A_star[center,i]=1
            A_star[i,center]=1
    graph_shifts=graph_shifts_helper(imgs,N,d,tries=tries,pos_tol=POS_TOL,status=status,A_fixed=A_star)
    status['total time']=rtime
    status['start']="rnd"
    status['mode']="Star shifts with center {}".format(center)
    save(graph_shifts,status)
    return graph_shifts

def blurr(imgs,xstd,ystd,size,status={}):
    """ Applies convolution to images. Preprocessing step to reduce noise. """
    status['kernel']="Gauss {},{},{}".format(xstd,ystd,size)
    kernel = tf.constant(xrf.gaussian_kernel(xstd,ystd,size),dtype='float32')
    blurred_imgs=[]
    for img in imgs:
        blurred_imgs.append(xrf.apply_convolution(img,kernel))    
    return blurred_imgs

if __name__ == "__main__":
    # execute only if run as a script
    
    # Example call: python compute_shifts 96 -ker 5 5 7 K 

    measure_id = sys.argv[1]
    if sys.argv[2] == '-ker':
        USE_KERNEL = True
        KERNEL_PARAM = np.array(sys.argv[3:6],dtype='int32')
        elements = sys.argv[6:]  # ['Ca']
    else:
        USE_KERNEL = False
        KERNEL_PARAM = None
        elements = sys.argv[2:]  

    do_direct_registration_shift_with_zero_start = True
    number_direct_registration_shift_with_random_start = 1
    do_complete_pw_shift =  True
    do_graph_shift_with_degrees =  [4,10,20]
    path_and_star = True
    pre_status={"measure_id":measure_id}

    MAX_IMAGES = None
    if MAX_IMAGES != None:
        pre_status["number of imgs"]=MAX_IMAGES

    for element in elements:
        pre_status["element"]=element
        imgs=read_data(measure_id,element)
        if MAX_IMAGES != None:
            N = np.min([MAX_IMAGES,N])
            imgs = imgs[:N]

        if USE_KERNEL:
            imgs=blurr(imgs,KERNEL_PARAM[0],KERNEL_PARAM[1],KERNEL_PARAM[2],status=pre_status)
        print(len(imgs),'images loaded.')
        if do_complete_pw_shift:
            print('start pairwise shift')
            compute_pw_shifts(imgs,status=pre_status.copy())
        for degree in do_graph_shift_with_degrees:
            print('start graph with degree ',degree)
            compute_graph_shifts(imgs,degree,status=pre_status.copy())
        if path_and_star:
            for i in [0,len(imgs)-1]:
                compute_star_shifts(imgs,i,status=pre_status.copy())
            compute_path_shifts(imgs,status=pre_status.copy())
        if do_direct_registration_shift_with_zero_start:
            print('start direct registration zero start')
            compute_direct_registration_shifts(imgs,start="zero",status=pre_status.copy())
        for running_number in range(number_direct_registration_shift_with_random_start):
            print('start direct registration random start {}/{}'.format(running_number+1,
                                                                        number_direct_registration_shift_with_random_start))
            compute_direct_registration_shifts(imgs,start="rnd",status=pre_status.copy())






