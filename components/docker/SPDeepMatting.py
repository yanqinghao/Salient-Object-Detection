# coding=utf-8
from __future__ import absolute_import, print_function

import os
from suanpan.app import app
from suanpan.screenshots import screenshots
from suanpan.app.arguments import Int, Folder
import tensorflow as tf
import numpy as np
import imageio
from PIL import Image
from utils import composite4, rgba2rgb, get_all_files


@app.input(Folder(key="inputData1"))
@app.input(Folder(key="inputData2"))
@app.param(Int(key="__gpu", default=0))
@app.output(Folder(key="outputData"))
def SPDeepMatting(context):

    args = context.args

    g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph("./meta_graph/my-model.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./salience_model"))
        image_batch = tf.get_collection("image_batch")[0]
        pred_mattes = tf.get_collection("mask")[0]

        rgb_pths = get_all_files(args.inputData1)
        for rgb_pth in rgb_pths:
            if os.path.splitext(rgb_pth)[-1] not in [".png", ".jpg", ".jpeg"]:
                continue
            rgb = imageio.imread(rgb_pth)
            if rgb.shape[2] == 4:
                rgb = rgba2rgb(rgb)
            origin_shape = rgb.shape[:2]

            rgb = np.expand_dims(
                np.array(
                    Image.fromarray(rgb.astype(np.uint8)).resize((320, 320))
                ).astype(np.float32)
                - g_mean,
                0,
            )

            feed_dict = {image_batch: rgb}
            pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)

            final_alpha = np.array(
                Image.fromarray(np.squeeze(pred_alpha)).resize(origin_shape[::-1])
            )

            final_alpha = final_alpha.astype(np.float64) / np.max(final_alpha)
            final_alpha = 255 * final_alpha
            final_alpha = final_alpha.astype(np.uint8)
            screenshots.save(final_alpha)
            # imageio.imwrite(os.path.join(args.outputData, rgb_pth), final_alpha)
            rgb_raw = imageio.imread(rgb_pth)
            bg_pths = get_all_files(args.inputData2)
            for bg_pth in bg_pths:
                if os.path.splitext(bg_pth)[-1] not in [".png", ".jpg", ".jpeg"]:
                    continue
                bg_raw = imageio.imread(bg_pth)
                if (
                    bg_raw.shape[0] < rgb_raw.shape[0]
                    or bg_raw.shape[1] < rgb_raw.shape[1]
                ):
                    rgb_raw = (
                        np.array(
                            Image.fromarray(rgb_raw.astype(np.uint8)).resize(
                                (
                                    bg_raw.shape[0],
                                    int(
                                        rgb_raw.shape[1]
                                        * bg_raw.shape[0]
                                        / rgb_raw.shape[0]
                                    ),
                                )
                            )
                        )
                        if bg_raw.shape[0] / rgb_raw.shape[0]
                        < bg_raw.shape[1] / rgb_raw.shape[1]
                        else np.array(
                            Image.fromarray(rgb_raw.astype(np.uint8)).resize(
                                (
                                    int(
                                        rgb_raw.shape[0]
                                        * bg_raw.shape[1]
                                        / rgb_raw.shape[1]
                                    ),
                                    bg_raw.shape[1],
                                )
                            )
                        )
                    )
                    final_alpha = (
                        np.array(
                            Image.fromarray(final_alpha.astype(np.uint8)).resize(
                                (
                                    bg_raw.shape[0],
                                    int(
                                        rgb_raw.shape[1]
                                        * bg_raw.shape[0]
                                        / rgb_raw.shape[0]
                                    ),
                                )
                            )
                        )
                        if bg_raw.shape[0] / rgb_raw.shape[0]
                        < bg_raw.shape[1] / rgb_raw.shape[1]
                        else np.array(
                            Image.fromarray(rgb_raw.astype(np.uint8)).resize(
                                (
                                    int(
                                        rgb_raw.shape[0]
                                        * bg_raw.shape[1]
                                        / rgb_raw.shape[1]
                                    ),
                                    bg_raw.shape[1],
                                )
                            )
                        )
                    )
                    origin_shape = (
                        (
                            int(rgb_raw.shape[1] * bg_raw.shape[0] / rgb_raw.shape[0]),
                            bg_raw.shape[0],
                        )
                        if bg_raw.shape[0] / rgb_raw.shape[0]
                        < bg_raw.shape[1] / rgb_raw.shape[1]
                        else (
                            bg_raw.shape[1],
                            int(rgb_raw.shape[0] * bg_raw.shape[1] / rgb_raw.shape[1]),
                        )
                    )
                im, bg = composite4(
                    rgb_raw, bg_raw, final_alpha, origin_shape[1], origin_shape[0]
                )
                if not os.path.exists(
                    os.path.join(
                        args.outputData, os.path.split(bg_pth)[1].split(".")[0]
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            args.outputData, os.path.split(bg_pth)[1].split(".")[0]
                        )
                    )
                imageio.imwrite(
                    os.path.join(
                        args.outputData,
                        os.path.split(bg_pth)[1].split(".")[0],
                        os.path.split(rgb_pth)[1],
                    ),
                    im,
                )
                screenshots.save(bg)
                # imageio.imwrite(os.path.join(args.outputData, "new_bg.png"), bg)

    return args.outputData
