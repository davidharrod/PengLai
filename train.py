from enum import Flag
import os
from re import L
import time
import numpy as np
from numpy.core.defchararray import mod
from numpy.lib.npyio import save
from pyntcloud.io import obj
from sonnet.python.modules.util import NOT_SUPPORTED
from pre_process import point_cloud_2_voxel as p2v
from tensor2tensor.layers.common_layers import batch_dense
import modules
import data_utils
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages

VERTICES = "VERTICES_FLAT"
FACES = "FACES"
EPOCH = 4
CHECK_POINT = 1
TRAIN_FIRST_TIME = "TRAIN_FIRST_TIME"
CONTINUE_TRAINING = "CONTINUE_TRAINING"
VISUALIZE = "VISUALIZE"

# Set tensorflow gpu configuration.
tf_gpu_config = tf.ConfigProto(allow_soft_placement=True)
tf_gpu_config.gpu_options.allow_growth = True

vertex_model_encoder_config = {
    "hidden_sizes": (64, 256),
    "num_blocks": (1, 2),
    "dropout_rate": 0.1
}
vertex_model_decoder_config = {
    "hidden_size": 256,
    "fc_size": 1024,
    "num_heads": 4,
    "dropout_rate": 0.2,
}
face_model_encoder_config = {
    'hidden_size': 128,
    'fc_size': 512,
    'num_layers': 3,
    'dropout_rate': 0.
}
face_model_decoder_config = {
    'hidden_size': 128,
    'fc_size': 512,
    'num_layers': 3,
    'dropout_rate': 0.
}


def _load_data(obj_path, binvox_path):
    ex_list = []
    obj_file_list = p2v._get_file_list(obj_path, file_type="OBJ")
    binvox_file_list = p2v._get_file_list(binvox_path, file_type="BINVOX")
    for i, file in enumerate(obj_file_list):
        mesh_dict = data_utils.load_process_mesh(file)
        mesh_dict["voxels"] = np.asarray(
            p2v.get_binarized_voxel_vector(binvox_file_list[i],
                                           file_type="BINVOX"))
        ex_list.append(mesh_dict)
    dataset = tf.data.Dataset.from_generator(
        lambda: ex_list,
        output_types={
            "vertices": tf.int32,
            "faces": tf.int32,
            "voxels": tf.int32
        },
        output_shapes={
            "vertices": tf.TensorShape([None, 3]),
            "faces": tf.TensorShape([None]),
            "voxels": mesh_dict["voxels"].shape  # todo: Determine this param.
        })
    ex = dataset.make_one_shot_iterator().get_next()
    return dataset, ex


def _make_model_dataset(dataset,
                        batch_size,
                        buffer_size,
                        apply_random_shift=False,
                        model_type=VERTICES):
    # todo: Fix dim bug.
    """Prepare the dataset for vertex model training with voxel condition."""
    if model_type == VERTICES:
        dataset = data_utils.make_vertex_model_dataset(
            dataset, apply_random_shift=apply_random_shift)
    elif model_type == FACES:
        dataset = data_utils.make_face_model_dataset(
            dataset, apply_random_shift=apply_random_shift)
    else:
        raise AttributeError(
            f"There are no {model_type} model in this package. Please try VERTEX or FACE instead."
        )
    dataset = dataset.repeat()
    dataset = dataset.shuffle(20).padded_batch(
        batch_size=batch_size, padded_shapes=dataset.output_shapes)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    return dataset  # batch是一个迭代器


def load_dataset(obj_path, binvox_path, batch_size, buffer_size):
    dataset, _ = _load_data(obj_path, binvox_path)
    vertex_dataset = _make_model_dataset(dataset,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         model_type=VERTICES)
    face_dataset = _make_model_dataset(dataset,
                                       batch_size=batch_size,
                                       buffer_size=buffer_size,
                                       model_type=FACES)
    return vertex_dataset, face_dataset


def _create_model(batch,
                  model_type,
                  encoder_config,
                  decoder_config,
                  quantization_bits=8,
                  max_sample_length=200,
                  max_seq_length=5000,
                  vertex_samples=None):
    """Return distribution predicted by the model, loss and samples."""
    if model_type == VERTICES:
        model = modules.VoxelToVertexModel(res_net_config=encoder_config,
                                           decoder_config=decoder_config,
                                           quantization_bits=quantization_bits)
        context = batch
    else:
        model = modules.FaceModel(class_conditional=False,
                                  encoder_config=encoder_config,
                                  decoder_config=decoder_config,
                                  max_seq_length=max_seq_length,
                                  quantization_bits=quantization_bits,
                                  decoder_cross_attention=True,
                                  use_discrete_vertex_embeddings=True)
        if vertex_samples:
            context = vertex_samples
        else:
            raise ValueError(
                "In order to create face model, you need to create vertex model first."
            )
    dist = model._build(batch)
    model_loss = -tf.reduce_sum(
        dist.log_prob(batch[f"{model_type.lower()}"]) *
        batch[f"{model_type.lower()}_mask"])
    if model_type == VERTICES:
        samples = model.sample(num_samples=4,
                               context=context,
                               max_sample_length=max_sample_length,
                               top_k=0,
                               top_p=0.95,
                               only_return_complete=False)
    else:
        samples = model.sample(context=context,
                               max_sample_length=max_sample_length,
                               top_k=0,
                               top_p=0.95,
                               only_return_complete=False)
    return dist, model_loss, samples


def _setup_training(vertex_dataset,
                    face_dataset,
                    learning_rate,
                    target_dir=None,
                    mode=None):
    # Prepare for training.
    vertex_batch = vertex_dataset.make_one_shot_iterator().get_next()
    face_batch = face_dataset.make_one_shot_iterator().get_next()
    _, vertex_loss, vertex_samples = _create_model(
        vertex_batch,
        model_type=VERTICES,
        encoder_config=vertex_model_encoder_config,
        decoder_config=vertex_model_decoder_config)
    _, face_loss, face_samples = _create_model(
        face_batch,
        model_type=FACES,
        encoder_config=face_model_encoder_config,
        decoder_config=face_model_decoder_config,
        vertex_samples=vertex_samples)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    vertex_model_optim_op = optimizer.minimize(vertex_loss)
    face_model_optim_op = optimizer.minimize(face_loss)
    # Set saver and summary.
    saver = tf.train.Saver()
    tf.summary.scalar("vertex loss", vertex_loss)
    tf.summary.scalar("face loss", face_loss)
    merged = tf.summary.merge_all()
    if mode == TRAIN_FIRST_TIME or mode == CONTINUE_TRAINING:
        if mode == TRAIN_FIRST_TIME:
            # Create directory for saving log and model.
            current_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
            writer_path = p2v._try_2_create_directory(
                target_dir, f"{current_time}/tensorboard/graphs")
            saver_path = p2v._try_2_create_directory(
                target_dir, f"{current_time}/ckpt_model")
        return vertex_model_optim_op, face_model_optim_op, writer_path, saver_path, saver, merged
    elif mode == VISUALIZE:
        return vertex_samples, face_samples, vertex_batch
    else:
        raise AttributeError(
            f"{mode} is not supported. Please try {CONTINUE_TRAINING} or {VISUALIZE} instead."
        )


def train(target_dir,
          vertex_dataset,
          face_dataset,
          learning_rate,
          training_step,
          check_step,
          mode=TRAIN_FIRST_TIME):
    # Setup training.
    vertex_model_optim_op, face_model_optim_op, writer_path, saver_path, saver, merged = _setup_training(
        vertex_dataset, face_dataset, learning_rate, target_dir, mode)
    # Training loop.
    with tf.Session(config=tf_gpu_config) as sess:
        writer = tf.summary.FileWriter(writer_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCH):
            for n in range(training_step):
                if n % check_step == 0:
                    summary = sess.run(merged)
                    writer.add_summary(summary, n + epoch * training_step)
                sess.run((vertex_model_optim_op, face_model_optim_op))
            if (epoch + 1) % CHECK_POINT == 0:
                saver.save(sess,
                           os.path.join(saver_path, "model.ckpt"),
                           global_step=epoch)
    return None


def continue_training(ckpt_dir, vertex_dataset, face_dataset, learning_rate,
                      training_step, check_step, epoch_remain, mode):
    """This function can be used to continue training as well as test model on validation dataset."""
    # Set up training.
    vertex_model_optim_op, face_model_optim_op, writer_path, saver_path, saver, merged = _setup_training(
        vertex_dataset, face_dataset, learning_rate)
    saver_restore = tf.train.Saver()
    with tf.Session(config=tf_gpu_config) as sess:
        # Load model.
        saver_restore.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        writer = tf.summary.FileWriter(writer_path, sess.graph)
        # Training loop.
        for epoch in range(epoch_remain):
            for n in range(training_step):
                if n % check_step == 0:
                    summary = sess.run(merged)
                    writer.add_summary(summary, n + epoch * training_step)
                if mode == CONTINUE_TRAINING:
                    sess.run((vertex_model_optim_op, face_model_optim_op))
                else:
                    pass
            if (epoch + 1) % CHECK_POINT == 0:
                saver.save(sess,
                           os.path.join(saver_path, "model.ckpt"),
                           global_step=epoch)
    return None


def visualize(ckpt_dir,
              vertex_dataset,
              face_dataset,
              learning_rate=5e-4,
              mode=VISUALIZE):
    vertex_samples, face_samples, vertex_batch = _setup_training(
        vertex_dataset, face_dataset, learning_rate, mode=mode)
    saver_restore = tf.train.Saver()
    with tf.Session() as sess:
        # Load model.
        saver_restore.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        vertex_samples_np, face_samples_np, _ = sess.run(
            (vertex_samples, face_samples, vertex_batch))
        mesh_list = []
        mesh_list.append({
            'vertices':
            vertex_samples_np['vertices'][0]
            [:vertex_samples_np['num_vertices'][0]],
            'faces':
            data_utils.unflatten_faces(
                face_samples_np['faces'][0]
                [:face_samples_np['num_face_indices'][0]])
        })
        data_utils.plot_meshes(mesh_list, ax_lims=0.5)
    return None


def _test_load_data(obj_path, binvox_path):
    """Create dataset successfully"""
    _, ex = _load_data(obj_path, binvox_path)
    # Inspect the first mesh
    with tf.Session() as sess:
        ex_np = sess.run(ex)
    print(ex_np)
    # Plot the meshes
    mesh_list = []
    with tf.Session() as sess:
        for i in range(1):
            ex_np = sess.run(ex)
            mesh_list.append({
                'vertices':
                data_utils.dequantize_verts(ex_np['vertices']),
                'faces':
                data_utils.unflatten_faces(ex_np['faces'])
            })
    data_utils.plot_meshes(mesh_list, ax_lims=0.4)
    return None


def _test_create_model(obj_path, binvox_path):
    """"""
    vertex_dataset, face_dataset = load_dataset(obj_path,
                                                binvox_path,
                                                batch_size=1,
                                                buffer_size=1)
    vertex_batch = vertex_dataset.make_one_shot_iterator().get_next()
    face_batch = face_dataset.make_one_shot_iterator().get_next()
    _, _, vertex_samples = _create_model(
        vertex_batch,
        model_type=VERTICES,
        encoder_config=vertex_model_encoder_config,
        decoder_config=vertex_model_decoder_config,
        max_sample_length=200)
    _, _, _ = _create_model(face_batch,
                            model_type=FACES,
                            encoder_config=face_model_encoder_config,
                            decoder_config=face_model_decoder_config,
                            max_seq_length=5000,
                            vertex_samples=vertex_samples)

    print("=====================================")
    print("Vertex model successfully created!")
    print("=====================================")
    print("Face model successfully created!")
    print("=====================================")
    return None


if __name__ == "__main__":
    obj_path = "/home/yqs/dave/data/polygen_data/train/poly"
    binvox_path = "/home/yqs/dave/data/polygen_data/train/binvox"
    target_dir = "./log/"
    ckpt_dir = "/home/yqs/dave/packages/PengLai/log/2022_01_07_18_34/ckpt_model"
    vertex_dataset, face_dataset = load_dataset(obj_path,
                                                binvox_path,
                                                batch_size=1,
                                                buffer_size=100)
    # Train.
    train(target_dir,
          vertex_dataset,
          face_dataset,
          learning_rate=5e-4,
          training_step=42408,
          check_step=2)

    # Visualize.
    # visualize(ckpt_dir, vertex_dataset, face_dataset)
