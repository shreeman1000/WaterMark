import subprocess
import sys
import pandas as pd
import io

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_stats = gpu_stats.decode('utf-8')
    gpu_df = pd.read_csv(io.StringIO(gpu_stats))
    gpu_df["memory.free"] = gpu_df[' memory.free [MiB]']
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]')).astype('float32')
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx
print(get_free_gpu()) 