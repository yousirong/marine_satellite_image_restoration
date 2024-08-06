import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

def rewrite_log_file(old_log_dir, new_log_dir):
    # Initialize the event accumulator for the old log file
    ea = event_accumulator.EventAccumulator(old_log_dir)
    ea.Reload()

    # Create a new SummaryWriter for the new log file
    writer = SummaryWriter(new_log_dir)

    # Rewrite Scalars
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        new_tag = tag.replace('Train/Loss', 'loss/train').replace('Validation/Loss', 'loss/validation')
        for event in events:
            writer.add_scalar(new_tag, event.value, event.step)

    writer.close()

# Example usage
old_log_dir = 'model/events.out.tfevents.1722600091.juneyonglee-RTX3090-2.1614283.0'  # 기존 이벤트 로그 파일 디렉토리 경로
new_log_dir = 'model/'  # 새로운 이벤트 로그 파일 디렉토리 경로

rewrite_log_file(old_log_dir, new_log_dir)
