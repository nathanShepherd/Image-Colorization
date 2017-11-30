#get local GPU device names!
#--> https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#Note:: that (at least up to TensorFlow 1.4), calling device_lib.list_local_devices()
# will run some initialization code that, by default, will allocate all
# of the GPU memory on all of the devices 

print(get_available_gpus())
