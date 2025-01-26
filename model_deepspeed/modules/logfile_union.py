import os
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf

def merge_event_files(log_dirs, output_dir, output_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = tf.summary.create_file_writer(output_dir)

    with writer.as_default():
        for log_dir in log_dirs:
            event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
            
            for event_file in event_files:
                print(f"Processing file: {event_file}")
                event_acc = event_accumulator.EventAccumulator(event_file)
                event_acc.Reload()

                for scalar in event_acc.Tags()['scalars']:
                    for event in event_acc.Scalars(scalar):
                        tf.summary.scalar(scalar, event.value, step=event.step)
                        writer.flush()

    print(f"New merged log file created at: {os.path.join(output_dir, output_file)}")

# Example usage
log_dirs = [
    'logs/ust_chl_4'
    # 추가 로그 디렉토리를 여기에 추가하세요
]
output_dir = 'logs/ust_chl_3'
output_file = 'merged_events.out.tfevents'

merge_event_files(log_dirs, output_dir, output_file)
