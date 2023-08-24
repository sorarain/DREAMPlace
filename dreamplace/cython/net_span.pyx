import cython
from cython.parallel cimport prange, parallel
import numpy as np
import numpy
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free

cpdef tuple node_pairs_among(int[:] nodes,int max_cap):
    cdef list us = []
    cdef list vs = []
    cdef int u
    cdef int v
    cdef int v_
    cdef int left
    if max_cap == -1 or len(nodes) <= max_cap:
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                us.append(u)
                vs.append(v)
    else:
        for u in nodes:
            vs_ = numpy.random.permutation(nodes)
            left = max_cap - 1
            if left == 0:
                continue
            for v_ in vs_:
                if left == 0:
                    break
                if u == v_:
                    continue
                us.append(u)
                vs.append(v_)
                left -= 1
    return (us, vs)

cdef double min(double* a,int len) nogil:
    cdef int i
    cdef double mn = 1e9
    for i in range(len):
        if(a[i] < mn):
            mn = a[i]
    return mn

cdef double max(double* a,int len) nogil:
    cdef int i
    cdef double mx = -1e9
    for i in range(len):
        if(a[i] > mx):
            mx = a[i]
    return mx

cdef int min_int(int* a,int len) nogil:
    cdef int i
    cdef int mn = 100000000
    for i in range(len):
        if(a[i] < mn):
            mn = a[i]
    return mn

cdef int max_int(int* a,int len) nogil:
    cdef int i
    cdef int mx = -100000000
    for i in range(len):
        if(a[i] > mx):
            mx = a[i]
    return mx

@cython.boundscheck(False)
@cython.wraparound(False)
def get_net_span(int num_nodes,int num_nets,
            int[:] flat_net2pin_map,int[:] flat_net2pin_start_map,
            int[:] pin2node_map,
            int[:] pin_offset_x,int[:] pin_offset_y,
            double[:,:] node_pos,
            double[:,:] net_span_feat,
            int bin_x,int bin_y):
    cdef int num_pin
    cdef int pin
    cdef int i
    cdef int net
    cdef int node
    cdef double x,y
    cdef double pin_px,pin_py
    cdef double px,py
    cdef int pinstart
    cdef double min_x,max_x,min_y,max_y
    cdef int min_px,max_px,min_py,max_py
    cdef double span_h,span_v,span_ph,span_pv

    with nogil:
        for net in prange(num_nets,schedule='guided',num_threads=64):
            pinstart = flat_net2pin_start_map[net]
            num_pin = flat_net2pin_start_map[net+1] - flat_net2pin_start_map[net]
            '''
            xs = <double*> malloc(sizeof(double) * num_pin)
            ys = <double*> malloc(sizeof(double) * num_pin)
            pxs = <int*> malloc(sizeof(int) * num_pin)
            pys = <int*> malloc(sizeof(int) * num_pin)
            '''
            min_x = 1e9
            max_x = -1e9
            min_y = 1e9
            max_y = -1e9
            min_px = 1000000000
            max_px = -1000000000
            min_py = 1000000000
            max_py = -1000000000
            
            for i in range(num_pin):
                pin = <int>(flat_net2pin_map[pinstart+i])
                node = int(pin2node_map[pin])
                x = node_pos[node,0]
                y = node_pos[node,1]
                
                pin_px,pin_py = pin_offset_x[pin],pin_offset_y[pin]
                px = x + pin_px
                py = y + pin_py

                '''
                xs[i] = px
                ys[i] = py
                pxs[i] = <int>(px / bin_x)
                pys[i] = <int>(py / bin_y)
                '''
                if min_x > px:
                    min_x = px
                if max_x < px:
                    max_x = px
                
                if min_y > py:
                    min_y = py
                if max_y < py:
                    max_y = py
                
                if min_px > <int>(px / bin_x):
                    min_px = <int>(px / bin_x)
                if max_px < <int>(px / bin_x):
                    max_px = <int>(px / bin_x)
                
                if min_py > <int>(py / bin_y):
                    min_py = <int>(py / bin_y)
                if max_py < <int>(py / bin_y):
                    max_py = <int>(py / bin_y)
            
            '''
            min_x,max_x,min_y,max_y = min(xs,num_pin),max(xs,num_pin),min(ys,num_pin),max(ys,num_pin)
            '''
            span_h = max_x - min_x + 1
            span_v = max_y - min_y + 1
            '''
            min_px,max_px,min_py,max_py = min_int(pxs,num_pin),max_int(pxs,num_pin),min_int(pys,num_pin),max_int(pys,num_pin)
            '''
            span_ph = max_px - min_px + 1
            span_pv = max_py - min_py + 1

            net_span_feat[net,0] = span_h
            net_span_feat[net,1] = span_v
            net_span_feat[net,2] = span_h * span_v
            net_span_feat[net,3] = span_ph
            net_span_feat[net,4] = span_pv
            net_span_feat[net,5] = span_ph * span_pv
            net_span_feat[net,6] = num_pin

    return net_span_feat
#end-def
