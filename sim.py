import time
import random
import multiprocessing as mp

# in ms
g_bandwidth = 1.0e6
g_latency = 0.24
g_nodes = 8
g_steps = 10
g_datasize = 50.0*1024*1024
g_test = False


def sleep(duration):
    time.sleep(duration/1000.0)

def test_redop(x1, x2):
    return x1 + x2

class Node:
    def __init__(self, idk, out_queue, next_q, prev_q):
        self.duration = 0.0
        self.idk = idk
        self.out_queue = out_queue
        self.next_q = next_q
        self.prev_q = prev_q

        self.iter_time = 175.0 + random.random()*50.0
        
        self.test_data = [ idk for _ in range(g_nodes) ]


    def timeSegmentTransfer(self):
        return g_latency + (g_datasize/g_nodes)/g_bandwidth

    def timeComputation(self):
        return self.iter_time -5 + 10*random.random()

    def send_nb(self, test_segid):
        self.next_q.put((self.duration, self.test_data[test_segid]))

#        self.next_q.put(self.duration)

    def receive_and_reduce_b(self, test_segid):
        duration, test_piece = self.prev_q.get()

        delay = self.timeSegmentTransfer()
        sleep(delay)
        if self.duration < duration+delay:
            self.duration = duration + delay

#        self.duration = self.prev_q.get()

        self.test_data[test_segid] = test_redop(test_piece, self.test_data[test_segid])

    def receive_b(self, test_segid):
        duration, test_piece = self.prev_q.get()
        
        delay = self.timeSegmentTransfer()
        sleep(delay)
        if self.duration < duration+delay:
            self.duration = duration + delay

#        self.duration = self.prev_q.get()
 
        self.test_data[test_segid] = test_piece

    def all_reduce_ring(self):
        # reudce
        self.send_nb(self.idk)
        for it in range(g_nodes-1):
            segment_id = (self.idk+g_nodes-1-it) % g_nodes
            self.receive_and_reduce_b(segment_id)
            self.send_nb(segment_id)
        # gather
        for it in range(g_nodes):
            segment_id = (self.idk+g_nodes-1-it) % g_nodes
            self.receive_b(segment_id)
            if it != g_nodes-1:
                self.send_nb(segment_id)

    def run(self):
        self.f = open("des0"+str(self.idk), "w")
        start = time.time()
        for i in range(g_steps):
            # computation phase
            delay = self.timeComputation()
            sleep(delay)
            self.duration += delay
            comp_ts = self.duration

            # communication phase
            self.all_reduce_ring()

            # save
            self.f.write("{} {}\n".format(comp_ts, self.duration-comp_ts))
 
            # test if alogrithms are correctly implemented
            if i == 0 and g_test:
                print(self.idk, self.test_data)

        self.out_queue.put((self.idk, time.time() - start, self.duration/1000.0))
        self.f.close()


if __name__ == '__main__':

    ns = []
    out_queue = mp.Queue()
    qs = [ mp.Queue() for _ in range(g_nodes) ] 
    next_qs = qs
    prev_qs = [qs[-1]] + qs[:-1]

    for it in range(g_nodes):
        n = Node(it, out_queue, next_qs[it], prev_qs[it])
        n.proc = mp.Process(target=n.run)
        n.proc.start()
        ns.append(n)

    for n in ns:
        n.proc.join()
        print(out_queue.get())
