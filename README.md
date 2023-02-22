# PPOthesis

# Singularity container
download .sif file of the tensorflow image
```
singularity pull docker://tensorflow/tensorflow:2.10.0-gpu
```

*If your host system has an NVIDIA GPU card and a driver installed, you can leverage the card with the `--nv` option*

run interactive node with GPUs
```
srun -p amdgpufast --gres=gpu:1 --pty bash -i
```

run the singularity image
```
cd /mnt/personal/sykorvo1/PPOthesis/ppo; singularity run --bind /mnt/personal/sykorvo1:/mnt/personal/sykorvo1 --nv tensorflow_2.10.0-gpu.sif
```

and now you have the terminal to run anything you like. For example
```
python run_model.py
```