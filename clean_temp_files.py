import os
# input_dir = '/home/smlm-workstation/event-smlm/Paris/MT_CL/'
input_dir = '/home/smlm-workstation/event-smlm/Paris/CL/'
# input_dir = '/home/smlm-workstation/event-smlm/Paris/MT/'

for dir in os.listdir(input_dir):
    if os.path.isdir(input_dir + dir):
        for f in os.listdir(input_dir + dir + '/temp_files/'):
            if f.startswith("localizations") or f.startswith("rois"):
                os.remove(input_dir + dir + '/temp_files/' + f)