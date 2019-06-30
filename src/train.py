import os
import time

import tensorflow as tf

from utils.options import Options
from data.dataset import Dataset
from models.cyclegan import CycleGANModel

tf.enable_eager_execution()
"""
Run this module for training.
Required args: --data_dir, --save_dir
"""
if __name__ == "__main__":
    opt = Options().parse(training=True)
    dataset = Dataset(opt)
    model = CycleGANModel(opt)

    device = ("/gpu:" + str(opt.gpu_id)) if opt.gpu_id != -1 else "/cpu:0"

    with tf.device(device):
        global_step = model.global_step
        batches_per_epoch = dataset.get_batches_per_epoch(opt)
        # Initialize Tensorboard summary writer:
        log_dir = os.path.join(opt.save_dir, 'tensorboard')
        summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
        for epoch in range(1, opt.epochs):
            start = time.time()
                for train_step in range(batches_per_epoch):
                    # Record summaries every 100 train_steps, we multiply by 3 because there are 3 gradient updates per step.
                    with summary_writer.as_default(),
                    tf.contrib.summary.record_summaries_every_n_global_steps(opt.summary_freq * 3, global_step=global_step):
                        model.set_input(dataset.data)
                        model.optimize_parameters()
                        if opt.save_summaries:
                            # Summaries for Tensorboard:
                            tf.contrib.summary.scalar('loss/genA2B', model.genA2B_loss)
                            tf.contrib.summary.scalar('loss/genB2A', model.genB2A_loss)
                            tf.contrib.summary.scalar('loss/discA', model.discA_loss)
                            tf.contrib.summary.scalar('loss/discB', model.discB_loss)
                            tf.contrib.summary.scalar('loss/cyc', model.cyc_lossA + model.cyc_lossB)
                            tf.contrib.summary.scalar('loss/identity', model.id_lossA + model.id_lossB)
                            tf.contrib.summary.scalar('learning_rate', model.learning_rate)
                            tf.contrib.summary.image('A/generated', model.fakeA)
                            tf.contrib.summary.image('A/reconstructed', model.reconstructedA)
                            tf.contrib.summary.image('B/generated', model.fakeB)
                            tf.contrib.summary.image('B/reconstructed', model.reconstructedB)
                        print("Iteration complete")
            # Assign decayed learning rate:
            model.update_learning_rate(batches_per_epoch)
            # Checkpoint the model:
            if epoch % opt.save_epoch_freq == 0:
                model.save_model()
            print("Global Training Step: ", global_step.numpy() // 3)
            print("Time taken for total epoch {} is {} sec\n".format(global_step.numpy() \
                                            // (3 * batches_per_epoch), time.time()-start))
