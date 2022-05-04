import os
import glob
import pickle
import sys
import gc
import re

import prep

from learning import LearningAgent
from sampler import SampleManager
import projects

import numpy as np

# Xception (ImageNet weights) has 3 channels
IMG_SHAPE = (128, 128, 3)

# custom xception bayes _bw needs 1 ch
#IMG_SHAPE = (128, 128, 1)

REGRESSION = 0

TRAIN_VAL_FRACTION = 0.85

SAVE_PICKLE = 0
SUMMARIZE = 0


def predict(xs, ys, keys, learner, sample_mapping, target_classes, target_class_converter, project_path, out_path, batch_idx):
    print("--------- predict samples ---------")
    # conv B+W
    if 0:
        xs = prep.convert_gray(xs)

    sample_classes, sample_split_key, sample_class_converter = sample_mapping(
        keys, project_path)

    y_labels = prep.make_labels(ys, keys, sample_class_converter)

    ys = prep.hot_ys(y_labels, target_classes)

    probys = learner.predict(xs)
    if SAVE_PICKLE:
        if batch_idx is None:
            ext = ".pickle"
        else:
            ext = "-{}.pickle".format(batch_idx)
        out_prob_path = out_path.replace(".csv", ext)

        # if agg for Bayes, dump all as pickle and take mean for csv
        if len(probys.shape) == 3:
            with open(out_prob_path, 'wb') as f:
                pickle.dump([keys, ys, probys], f)

    if len(probys.shape) == 3:
        probys = np.mean(probys.numpy(), axis=0)

    prep.dump_prediction(out_path, keys, probys, y_labels, append=True)

    if SUMMARIZE:
        pys = prep.onehot(probys)
        try:
            prep.summarize(ys, pys, sample_classes)

            real_y_labels = prep.make_labels(ys, keys, target_class_converter)
            real_ys = prep.hot_ys(real_y_labels, target_classes)
            if real_ys is None:
                print("Target doesn't map.  Unknown accuracy")
            else:
                prep.acc(real_ys, pys)

        except:
            print("Summarize Error!!")


def main(project, model_path, img_path, img_name_pattern, out_path, action):
    pos = img_path.rfind("/")
    project_path = img_path[0:pos + 1]
    out_path = project_path + out_path

    pos = project.find(",")
    project_target = project[0:pos]
    project_sample = project[pos + 1:]

    target_mapping = eval("projects.prepare_" + project_target)
    target_classes, target_split_key, target_class_converter = target_mapping(
        None, project_path)
    sample_mapping = eval("projects.prepare_" + project_sample)

    use_all_samples = False
    if "!" in img_path:
        img_path = img_path.replace("!", "")
        use_all_samples = True

    merge_path = img_path.replace("*", "merged")

    auto_merge = ("*" in img_path)

    train_img_path = merge_path.replace(".pickle", "-train.pickle")
    val_img_path = merge_path.replace(".pickle", "-val.pickle")

    learner = LearningAgent(model_path, img_shape=IMG_SHAPE,
                            target_classes=target_classes)

    if action == "prep":
        sampler = SampleManager(merge_path)

        if auto_merge:
            img_paths = glob.glob(img_path)
            for batch_idx, one_path in enumerate(img_paths):
                if "*" in one_path:
                    filter = img_path.replace("*", "\\d+")
                    if not re.search(filter, one_path):
                        # skip
                        continue

                one_sampler = SampleManager(one_path)
                sampler.merge(one_sampler)

        train_sampler = SampleManager(train_img_path)
        val_sampler = SampleManager(val_img_path)
        if train_sampler.count() > 0:
            print("********* Train sampler has data.  Skip creation")
            quit()
        if val_sampler.count() > 0:
            print("********* Val sampler has data.  Skip creation")
            quit()

        xs, ys, keys = sampler.get()

        xs = list(xs)
        keys = list(keys)
        xs = prep.fix_sizes(xs, IMG_SHAPE)

        # keep here.  need in train as well...
        ys = prep.make_labels(ys, keys, target_class_converter)

        if target_split_key is None:
            train_xs, train_ys, train_keys, val_xs, val_ys, val_keys = prep.split3(
                xs, ys, keys, TRAIN_VAL_FRACTION)
        else:
            train_xs, train_ys, train_keys, val_xs, val_ys, val_keys = prep.split3_by_key(
                xs, ys, keys, TRAIN_VAL_FRACTION, target_split_key, keep_min=1000)

        for idx, x in enumerate(train_xs):
            y = train_ys[idx]
            k = train_keys[idx]
            train_sampler.add(x, y, k)
        train_sampler.save_samples()

        for idx, x in enumerate(val_xs):
            y = val_ys[idx]
            k = val_keys[idx]
            val_sampler.add(x, y, k)
        val_sampler.save_samples()

        print("Created training/val sets!")

    elif action == "train":
        train_sampler = SampleManager(train_img_path)
        val_sampler = SampleManager(val_img_path)

        train_xs, train_ys, train_keys = train_sampler.get()
        val_xs, val_ys, val_keys = val_sampler.get()

        train_xs = prep.fix_channels(train_xs, IMG_SHAPE)
        val_xs = prep.fix_channels(val_xs, IMG_SHAPE)

        train_ys = prep.make_labels(
            train_ys, train_keys, target_class_converter)
        val_ys = prep.make_labels(val_ys, val_keys, target_class_converter)

        train_xs, train_ys, train_keys = prep.remove_invalids(
            train_xs, train_ys, train_keys, check_y=True)
        val_xs, val_ys, val_keys = prep.remove_invalids(
            val_xs, val_ys, val_keys, check_y=True)

        for label in target_classes:
            print(label, train_ys.count(label), '/', val_ys.count(label))

        ox, oy, ok = val_xs.copy(), val_ys.copy(), val_keys.copy()

        val_y_labels = val_ys
        if not REGRESSION:
            train_ys = prep.hot_ys(train_ys, target_classes)
            val_y_labels = val_ys
            val_ys = prep.hot_ys(val_y_labels, target_classes)

        learner.train(train_xs, train_ys, val_xs, val_ys)

        val_xs, val_ys, val_keys = ox, oy, ok
        predict(val_xs, val_ys, val_keys, learner, sample_mapping,
                target_classes, target_class_converter, project_path, out_path, None)

    elif action == "predict":
        if not use_all_samples and os.path.exists(val_img_path):
            print("Using VALIDATION data only!")
            predict_img_path = val_img_path
            do_fix_sizes = False

        else:
            print("Validation data not available, using FULL data set!")
            predict_img_path = img_path
            do_fix_sizes = True

        if os.path.exists(out_path):
            os.remove(out_path)

        predict_paths = glob.glob(predict_img_path)
        for batch_idx, predict_path in enumerate(predict_paths):
            if "*" in predict_img_path:
                filter = predict_img_path.replace("*", "\\d+")
                if not re.search(filter, predict_path):
                    # skip
                    continue

            val_sampler = SampleManager(predict_path)
            val_xs, val_ys, val_keys = val_sampler.get()

            if do_fix_sizes:
                # done automatically for validation data, do manually here
                val_xs = prep.fix_sizes(val_xs, IMG_SHAPE)

            val_xs = prep.fix_channels(val_xs, IMG_SHAPE)

            predict(val_xs, val_ys, val_keys, learner, sample_mapping, target_classes,
                    target_class_converter, project_path, out_path, batch_idx)

            del val_sampler, val_xs, val_ys, val_keys
            gc.collect()

    else:
        print("Action not recognized.  NO OP")


if __name__ == "__main__":
    _, project, model_path, img_path, img_name_pattern, out_path, action = sys.argv

    print()
    print("Starting Classifier Main:")
    print(sys.argv)

    main(project, model_path, img_path, img_name_pattern, out_path, action)
