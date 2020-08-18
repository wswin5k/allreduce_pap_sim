import time
import random
import numpy as np
import multiprocessing as mp
import sys
import os

np.random.seed(1234)

MEBI = 1024*1024
GIBI = 1024*MEBI
MEGA = 1e6
GIGA = 1e9

# config
g_rootlog = '/macierz/home/s160690/masters/examples-dist/meas/erty-sim/resnet50/ring/ranks8/bs1024/4/'
g_latency_avg = 0.24886
g_datasize = 25_557_032*4
g_datasize = MEBI*4
g_bandwidth_avg = [ [779.30, 48.49, 48.49, 96.85],
                    [48.49, 780.66, 96.92, 48.49],
                    [48.49, 96.87, 780.86, 48.49],
                    [96.88, 48.49, 48.48, 779.89]]
g_bandwidth_avg = np.array(g_bandwidth_avg)*GIBI/1000
g_bandwidth_avg = 808.56*MEGA/8/1000

def test_redop(x1, x2):
    return x1 + x2

class DurationModel:
    def __init__(self):
        pass

    def init_tau(self):
        mtrcs = []
        for _ in range(100000):
            src = np.random.randint(self.P)
            dst = (src+1)%self.P
            mtrcs.append(self.timeSegmentTransfer(src, dst))
        self.tau = np.mean(mtrcs)

    @classmethod
    def createFromFile(cls, fname):
        dm = DurationModel()
        with open(fname) as f:
            data = np.loadtxt(fname)
            dm.dataComp = []
            dm.len = data.shape[1]
            for seq in data:
                dm.dataComp.append(iter(seq))
        dm.P = data.shape[0]
        dm.init_tau()
        return dm

    @classmethod
    def createOneRandomDist(cls, P, n, delay, maxDelay):
        dm = DurationModel()
        dm.P = P
        dm.dataComp = []
        for it in range(P):
            if it == 1:
                dm.dataComp.append(iter(list(np.full((n,), maxDelay+delay))))
            else:
                dm.dataComp.append(iter(list(np.full((n,), delay))))
        dm.init_tau()
        dm.len = n
        return dm

    def timeSegmentTransfer(self, src, dst):
        if type(g_bandwidth_avg) == float:
            bandwidth = g_bandwidth_avg
        else:
            bandwidth = g_bandwidth_avg[src][dst]
        latency = g_latency_avg
        return latency + (g_datasize/self.P)/bandwidth

    def timeComputation(self, node):
        return next(self.dataComp[node])

    def __len__(self):
        return self.len

class Node:
    def __init__(self, idk, cqueues, dm, arrivals, arrivalsBarrier, idsMap, prr):
        self.dm = dm
        self.duration = 0.0
        self.idk = idk
        self.idk_init = idk
        self.ops = 0
        self.cqueues = cqueues
        self.prr = prr

        self.arrivals = arrivals
        self.arrivalsBarrier = arrivalsBarrier
        self.idsMap = idsMap

        if idk == 0: print(self.dm.tau)
 
        self.test_data = [ idk+1 for _ in range(self.dm.P) ]

        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        self.update_queues()

    def update_queues(self):
        self.next_q = self.cqueues[self.idk][(self.idk+1)%self.dm.P]
        self.prev_q = self.cqueues[(self.idk+self.dm.P-1)%self.dm.P][self.idk]

### Communication primitives
    def send_nb(self, test_segid):
        self.next_q.put((self.duration, self.test_data[test_segid]))

    def durate(self, duration, delay):
        if self.duration < duration:
            self.duration = duration + delay
        else:
            self.duration += delay
        return 
        if self.duration < duration+duration:
            self.duration = duration + delay
        return

    def receive_and_reduce_b(self, test_segid):
        duration, test_piece = self.prev_q.get()

        src = self.idsMap[(self.idk+self.dm.P-1)%self.dm.P]
        delay = self.dm.timeSegmentTransfer(src, self.idk_init)
        self.durate(duration, delay)
        self.ops += 1

        self.test_data[test_segid] = test_redop(test_piece, self.test_data[test_segid])

    def receive_b(self, test_segid):
        duration, test_piece = self.prev_q.get()
        
        src = self.idsMap[(self.idk+self.dm.P-1)%self.dm.P]
        delay = self.dm.timeSegmentTransfer(src, self.idk_init)
        self.durate(duration, delay)
        self.ops += 1
        
        self.test_data[test_segid] = test_piece

### All-reduce algorithms
    def all_reduce_pre_reduced_ring(self, prr=False):
        n = self.dm.P
        if prr:
            # get arrivals of all processes
            self.arrivals[self.idk] = self.duration
            self.arrivalsBarrier.wait()
            
            # calculate new id and update communications queues
            idks_sorted = np.argsort(np.argsort(self.arrivals))
            self.idk = idks_sorted[self.idk]
            self.idsMap[self.idk] = self.idk_init
            self.arrivalsBarrier.wait()
            if self.idk == 0:
                self.arrivals.sort()
            self.arrivalsBarrier.wait()
            self.update_queues()
            self.arrivalsBarrier.wait()

            # calculate numer of segments for early processes
            k = [0]*n
            for it in range(n-2, -1, -1):
                if self.arrivals[n-1] - self.arrivals[it+1] > self.dm.tau*(k[it+1]+1):
                    k[it] = k[it+1] + 1
                else:
                    k[it] = k[it+1]
            startProcForSeg = [ 0 for i in range(n) ]
            lastProcForSeg = [ 0 for i in range(n) ]
            proc_it = 0
            for seg_it in range(n):
                while k[proc_it]+proc_it < seg_it:
                    proc_it += 1
                startProcForSeg[seg_it] = proc_it
                lastProcForSeg[seg_it] = (startProcForSeg[seg_it]+n-1) % n
            '''
            proc_it = 0
            seg_it = 0
            while seg_it < n:
                if proc_it+k[proc_it] >= seg_it:
                    startProcForSeg[seg_it] = proc_it
                    lastProcForSeg[seg_it] = (startProcForSeg[seg_it]+n-1) % n
                    seg_it += 1
                else:
                    proc_it += 1
            '''
            # reudce
            segment_id = self.idk + k[self.idk] #starting segment
        else:
            startProcForSeg = np.arange(n)
            lastProcForSeg = (startProcForSeg+n-1)%n
            segment_id = self.idk
        for _ in range(n):
            if startProcForSeg[segment_id] != self.idk:
                self.receive_and_reduce_b(segment_id)
            self.send_nb(segment_id)
            segment_id = (segment_id+n-1) % n
        # gather
        for _ in range(n):
            if lastProcForSeg[segment_id] != self.idk:
                self.receive_b(segment_id)
                if ((lastProcForSeg[segment_id]+n-1) % n) != self.idk:
                    self.send_nb(segment_id)
            segment_id = (segment_id+n-1) % n

    def all_reduce_ring(self):
        # reudce
        segment_id = self.idk
        start = self.duration
        self.send_nb(segment_id)
        for it in range(self.dm.P-1):
            segment_id = (segment_id+self.dm.P-1) % self.dm.P
            self.receive_and_reduce_b(segment_id)
            self.send_nb(segment_id)
        # gather
        for it in range(self.dm.P-1):
            segment_id = (segment_id+self.dm.P-1) % self.dm.P
            self.receive_b(segment_id)
            if it != self.dm.P-2:
                self.send_nb(segment_id)

### Main program loop
    def run(self):
        self.f = open(g_rootlog+"sim0"+str(self.idk_init+1), "w")
        for i in range(len(self.dm)):
            # computation phase
            compStart = self.duration
            delay = self.dm.timeComputation(self.idk_init)
            self.duration += delay
            compEnd = self.duration

            # communication phase
            if self.prr:
                self.all_reduce_pre_reduced_ring(True)
            else:
                self.all_reduce_ring()
            commEnd = self.duration
            # save
            self.f.write("{} {} {} {}\n".format(compStart/1000.0, 0.0, compEnd/1000.0, commEnd/1000.0))

            # test if alogrithms are correctly implemented
            assert [sum(range(1, self.dm.P+1)) for i in range(self.dm.P)] == self.test_data, "Reduction error."
            self.test_data = [ self.idk_init+1 for _ in range(self.dm.P) ]
            self.arrivalsBarrier.wait()
 
        print(self.idk_init, self.ops, self.duration)
        self.f.close()

def run(prr=True):
    ns = []
    #dm = DurationModel.createFromFile("1024.txt")
    dm = DurationModel.createOneRandomDist(30, 1000, 100, 100)
    qs = [ [ mp.Queue() for _ in range(dm.P) ] for _ in range(dm.P) ]
    next_qs = qs
    prev_qs = [qs[-1]] + qs[:-1]
    
    man = mp.Manager()
    arrivals = man.list(range(dm.P))
    idsMap = man.list(range(dm.P))
    arrivalsBarrier = mp.Barrier(dm.P)

    for it in range(dm.P):
        n = Node(it, qs, dm, arrivals, arrivalsBarrier, idsMap, prr)
        n.proc = mp.Process(target=n.run)
        n.proc.start()
        ns.append(n)

    for n in ns:
        n.proc.join()

if __name__ == '__main__':
    run(True)
    run(False)
