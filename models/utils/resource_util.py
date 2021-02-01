import os
import psutil
import nvsmi
import nvidia_smi
import time
import datetime
import inspect
import multiprocessing as mp
import threading


def resource_util(pid, interval):
    '''
        arg:
            pid: process id (int)
        
        example return:
            {
                'pid': 24832, 
                'cpu': 0.0, 
                'mem_total': 3371, 
                'mem_shared': 502, 
                'mem_data': 3039, 
                'gpu_id': 0, 
                'gpu_mem': 5985.0, 
                'gpu_usage': 100, 
                'result': [24832, 0.0, 3371, 502, 3039, 0, 5985.0, 100]
            }
    '''
    nvidia_smi.nvmlInit()
    # Get resources used by process
    p = psutil.Process(pid)
    usage = {'pid': pid}
    result = [pid]

    # cpu usage of current PID
    usage['cpu'] = p.cpu_percent(interval=interval)
    result.append(usage['cpu'])
    # Memory usage current PID
    mem = p.memory_info()
    # print(mem, type(mem))
    usage['mem_total'] = mem.rss >> 20
    result.append(usage['mem_total'])
    usage['mem_shared'] = mem.shared >> 20
    result.append(usage['mem_shared'])
    usage['mem_data'] = mem.data >> 20
    result.append(usage['mem_data'])
    
    for process in (nvsmi.get_gpu_processes()):
        # print(process.pid, process.gpu_id, process.used_memory)
        if process.pid == pid:
            usage['gpu_id'] = int(process.gpu_id)
            result.append(usage['gpu_id'])
            usage['gpu_mem'] = process.used_memory
            result.append(usage['gpu_mem'])
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(process.gpu_id))
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            usage['gpu_usage'] = res.gpu # gpu utilization, may not only by this process
            result.append(usage['gpu_usage'])
            break
    else:
        usage['gpu_id'] = None
        result.append(usage['gpu_id'])
        usage['gpu_mem'] = None
        result.append(usage['gpu_mem'])
        usage['gpu_usage'] = None # gpu utilization, may not only by this process
        result.append(usage['gpu_usage'])
    
    usage['result'] = result
    return usage


def start_monitor(path, note, pid=None, interval=0.1):

    """
        starts monitoring the resource usage of process

        args:
            path (str): the path of log directory 
            note (str): a custom note that is added to log result 
            pid  (int): process id default to the process that you called this function
            interval (float): measurement interval of cpu usage (must be >0)
        
        return:
            monitor (tuple): use this object in stop monitor to finish monitoring
        
        example usage: if we want to measure the resource usage of func

                ... some code

                print(abc)
                monitor = start_monitor(args.dout, "rollout step=10")
                func()
                stop_monitor(monitor)

                ... some more code

            stop monitor writes:
                38790,106.9,2871,526,2962,0,1129.0,2,2021-02-01 03:45:48.855156,1.164,run_rollouts,rollout step=10\n
            
            to args.dout/resource_monitor.csv

            which means:
                pid: 38790, function: run_rollouts        , note: rollout step=10, datetime: 2021-02-01 03:45:48.855156
                [cpu: 106.9, time_took: 1.164, mem_total: 2871, mem_shared: 526, mem_data: 2962,0, gpu_id: 0, gpu_mem: 1129.0, gpu_usage: 2]
        
        NOTE:
        use python ALFRED/resource_monitor.py exp/<dout folder>/resource_monitor.csv
            to view real time monitoring of current execution.

    """

    t0 = time.time()

    print(f"start monitor pid={pid}, path={path}, note={note}")

    if pid == None:
        pid = os.getpid()
    res = list()

    def monitor_process(res):
        for i in (resource_util(pid, interval)['result']):
            res.append(i)

    p = threading.Thread(target=monitor_process, args=(res,))
    p.start()
    return (res, p, t0, path, note)


def stop_monitor(measurement):
    res, p, t0, path, note = measurement
    print(f"stop monitor path={path}, note={note}")
    p.join()
    result = list(res) + [str(datetime.datetime.now()), str(time.time()-t0), inspect.stack()[1][3], note]
    outpath = '/'.join(os.path.split(path) + ('resource_monitor.csv',))
    with open(outpath, 'a') as f:
        f.write(','.join([str(i) for i in result])+'\n')
    # print(','.join([str(i) for i in result])+'\n')


if __name__ == '__main__':
    print("=========")
    m = mp.Manager()
    measure = start_measure(interval=0.1, pid=24252, manager=m)
    # time.sleep(0.1)
    stop_measure(measure, '', '')

    measure = start_measure(interval=0.1, pid=25300, manager=m)
    # time.sleep(0.1)
    stop_measure(measure, '', '')