import time
import random
import numpy as np
import multiprocessing as mp
import sys
import os

np.random.seed(42)

MEBI = 1024*1024
GIBI = 1024*MEBI
MEGA = 1e6
GIGA = 1e9

### Configuration
g_rootlog = 'log/'
g_compfile = 'comp.txt'
g_idsfile = None
g_prr = True
g_flops = 3855.3e9
g_directwrite = False # communication model
g_params = 25_557_032
g_datasize = g_params*4
g_latency_avg = 0.24886
g_bandwidth_avg = 808.56*MEGA/8/1000


def test_redop(x1, x2):
    return x1 + x2

class DurationModel:
    def __init__(self):
        pass

    def init_tau(self):
        mtrcs = []
        for _ in range(100000):
            src, dst = np.random.choice(range(self.P), 2)
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

    def timeReduction(self):
        return (g_params/self.P)/(g_flops/1000.0)

    def __len__(self):
        return self.len

class Node:
    def __init__(self, idk, cqueues, dm, arrivals, arrivalsBarrier, idsMap, new_ids_path=None):
        self.dm = dm
        self.duration = 0.0
        self.idk = idk
        self.idk_init = idk
        self.ops = 0
        self.cqueues = cqueues

        self.arrivals = arrivals
        self.arrivalsBarrier = arrivalsBarrier
        self.idsMap = idsMap

        if g_prr:
            self.algorithm = self.all_reduce_pre_reduced_ring
        else:
            self.algorithm = self.all_reduce_ring

        if g_directwrite:
            self.durate = self.durateRDMA
        else:
            self.durate = self.durateNormal

        if new_ids_path:
            self.new_ids = iter(np.loadtxt(new_ids_path))
            self.get_new_ids = self.new_ids_est
        else:
            self.get_new_ids = self.new_ids_real

        if idk == 0: print(self.dm.tau)
 
        self.test_data = [ idk+1 for _ in range(self.dm.P) ]

        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        self.update_queues()

    def update_queues(self):
        self.next_q = self.cqueues[self.idk][(self.idk+1)%self.dm.P]
        self.prev_q = self.cqueues[(self.idk+self.dm.P-1)%self.dm.P][self.idk]

### Communication primitives
    def durate(self, duration, delay):
        if self.duration < duration+delay:
            self.duration = duration + delay
        return
        if self.duration < duration:
            self.duration = duration + delay
        else:
            self.duration += delay
        return
#### cluster env
    def send_nb(self, test_segid):
        self.next_q.put((self.duration, self.test_data[test_segid]))

    def receive_and_reduce_b(self, test_segid):
        duration, test_piece = self.prev_q.get()

        src = self.idsMap[(self.idk+self.dm.P-1)%self.dm.P]
        delay = self.dm.timeSegmentTransfer(src, self.idk_init)
        self.durate(duration, delay)
        self.duration += self.dm.timeReduction()
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
    def new_ids_real(self):
        self.arrivals[self.idk] = self.duration
        self.arrivalsBarrier.wait()    
        idks_sorted = np.argsort(np.argsort(self.arrivals))
        self.idk = idks_sorted[self.idk]
        self.idsMap[self.idk] = self.idk_init
        self.arrivalsBarrier.wait()
        if self.idk == 0:
            self.arrivals.sort()

    def new_ids_est(self):
        idks_arrivals = next(self.new_ids)
        idks_sorted = np.argsort(np.argsort(self.arrivals))
        self.idk = idks_sorted[self.idk_init]
        self.idsMap[self.idk] = self.idk_init

    def all_reduce_pre_reduced_ring(self, prr=True):
        n = self.dm.P
        if prr:
            # get arrivals of all processes
            # calculate new id and update communications queues
            self.get_new_ids()
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
            # reudce
            segment_id = self.idk + k[self.idk] #starting segment
        else:
            startProcForSeg = np.arange(n)
            lastProcForSeg = (startProcForSeg+n-1)%n
            segment_id = self.idk
        for _ in range(n):
            if startProcForSeg[segment_id] != self.idk:
                self.receive_and_reduce(segment_id)
            self.send(segment_id)
            segment_id = (segment_id+n-1) % n
        # gather
        for _ in range(n):
            if lastProcForSeg[segment_id] != self.idk:
                self.receive(segment_id)
                if ((lastProcForSeg[segment_id]+n-1) % n) != self.idk:
                    self.send(segment_id)
            segment_id = (segment_id+n-1) % n

    def all_reduce_ring(self):
        # reudce
        segment_id = self.idk
        start = self.duration
        self.send(segment_id)
        for it in range(self.dm.P-1):
            segment_id = (segment_id+self.dm.P-1) % self.dm.P
            self.receive_and_reduce(segment_id)
            self.send(segment_id)
        # gather
        for it in range(self.dm.P-1):
            segment_id = (segment_id+self.dm.P-1) % self.dm.P
            self.receive(segment_id)
            if it != self.dm.P-2:
                self.send(segment_id)

### Main program loop
    def run(self):
        path = g_rootlog
        os.makedirs(path, exist_ok=True)
        self.f = open(path+"sim"+str(self.idk_init+1), "w")
        for i in range(len(self.dm)):
            # computation phase
            compStart = self.duration
            delay = self.dm.timeComputation(self.idk_init)
            self.duration += delay
            compEnd = self.duration

            # communication phase
            self.algorithm()
            commEnd = self.duration
            # save
            self.f.write("{} {} {} {}\n".format(compStart/1000.0, 0.0, compEnd/1000.0, commEnd/1000.0))

            # test if alogrithms are correctly implemented
            assert [sum(range(1, self.dm.P+1)) for i in range(self.dm.P)] == self.test_data, "Reduction error."
            self.test_data = [ self.idk_init+1 for _ in range(self.dm.P) ]
            self.arrivalsBarrier.wait()
 
        print(self.idk_init, self.ops, self.duration)
        self.f.close()

def run():
    ns = []
    dm = DurationModel.createFromFile(g_compfile)
    qs = [ [ mp.Queue() for _ in range(dm.P) ] for _ in range(dm.P) ]
    next_qs = qs
    prev_qs = [qs[-1]] + qs[:-1]
 
    man = mp.Manager()
    arrivals = man.list(range(dm.P))
    idsMap = man.list(range(dm.P))
    arrivalsBarrier = mp.Barrier(dm.P)

    for it in range(dm.P):
        n = Node(it, qs, dm, arrivals, arrivalsBarrier, idsMap, g_idsfile)
        n.proc = mp.Process(target=n.run)
        n.proc.start()
        ns.append(n)

    for n in ns:
        n.proc.join()

if __name__ == '__main__':
    run()
