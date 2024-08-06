import tensorflow as tf
from tensorflow.compat.v1.train import summary_iterator
import os

# Original .tfevents file path
log_dir = 'model'
original_file = os.path.join(log_dir, 'events.out.tfevents.1722600091.juneyonglee-RTX3090-2.1614283.0')

# New .tfevents file path
new_file = os.path.join(log_dir, 'events.out.tfevents.modified')

# Modify and filter event data
with tf.io.TFRecordWriter(new_file) as writer:
    for event in summary_iterator(original_file):
        if event.step == 388000:
            # Modify event as needed
            for value in event.summary.value:
                if value.tag == 'Train/Loss':
                    value.simple_value = 0.5429  # Modify to desired value
                if value.tag == 'Validation/Loss':
                    value.simple_value = 0.5147
        writer.write(event.SerializeToString())
