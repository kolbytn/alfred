import time
import subprocess
import select
import sys

if __name__ == '__main__':

    f = subprocess.Popen(['tail','-F',sys.argv[1]], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    p = select.poll()
    p.register(f.stdout)

    while True:
        if p.poll(1):
            info = (f.stdout.readline().decode('ascii').rstrip().split(','))
            
            pid, cpu, mem_total, mem_shared, mem_data, gpu_id, gpu_mem, gpu_usage, datetime, time_took, stack_location, note = info

            print(f"pid: {pid}, function: {stack_location:<20}, note: {note}, datetime: {datetime}\n[cpu: {cpu:<5}, time_took: {float(time_took):.3f}, mem_total: {mem_total}, mem_shared: {mem_shared}, mem_data: {mem_data}, gpu_id: {gpu_id}, gpu_mem: {gpu_mem}, gpu_usage: {gpu_usage}]\n")