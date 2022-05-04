
import glob
import imageio
import numpy as np
import csv
import re

from skimage import color, util
from skimage import exposure
from skimage import transform


def reshape_img(img):
    if len(img.shape) == 2:
        ni = np.empty(shape=(img.shape[0], img.shape[1], 1), dtype=img.dtype)
        ni[:, :, 0] = img
        return ni
    else:
        return img


def fix_sizes(xs, shape):
    # keep channels from original, only adjust w,h
    if len(xs[0].shape) == 3:
        ch = xs[0].shape[2]
    else:
        ch = 1
    shape = (shape[0], shape[1], ch)

    large_img_count = 0
    fixed_sizes = 0
    for idx, x in enumerate(xs):
        if x.shape != shape:
            img = None
            x = reshape_img(x)
            h, w, _ = x.shape
            if h <= shape[0] and w <= shape[1]:
                # paste into standard size, centered
                img = np.zeros(shape=shape, dtype=xs[0].dtype)
                y1 = (shape[0] - h) // 2
                x1 = (shape[1] - w) // 2
                y2 = y1 + h
                x2 = x1 + w
                img[y1:y2, x1:x2, :] = x[:, :, :]
                fixed_sizes += 1
            else:
                # raise Exception("x shape {} exceeds {}".format(x.shape, shape))
                large_img_count += 1

            xs[idx] = img

    if large_img_count > 0:
        print("fixing sizes but {} images were too large".format(large_img_count))

    if fixed_sizes > 0:
        print("fixing sizes: {}".format(fixed_sizes))
    else:
        print("*** no size fix.  must be perfect!!!")

    return xs


def make_labels(ys, keys, class_converter):
    labels = []
    for idx, key in enumerate(keys):
        y = ys[idx]
        labels.append(class_converter(y, key))
    return labels


def remove_invalids(xs, ys, keys, check_y=False):
    new_xs, new_ys, new_keys = [], [], []
    for idx, y in enumerate(ys):
        if xs[idx] is not None:
            if not check_y or y is not None:
                new_xs.append(xs[idx])
                new_ys.append(y)
                new_keys.append(keys[idx])

    if len(xs) != len(new_xs):
        print("removed invalids:", (len(xs) - len(new_xs)))
    return new_xs, new_ys, new_keys


def load_by_folder_label(path, name_pattern):

    files = glob.glob(path + name_pattern)
    print("Found files:", len(files))
    xs, ys, ks = [], [], []

    for fn in files:

        label = fn.replace(path, "")
        key = label

        img = imageio.imread(fn)

        xs.append(img)
        ys.append(label)
        ks.append(key)

    return xs, ys, ks


def one_hot(val, target_classes):
    ny = np.zeros(len(target_classes), dtype=np.uint8)

    idx = target_classes.index(val)
    ny[idx] = 1

    return ny


def hot_ys(ys, target_classes):
    yy = []

    log_limit = 10
    error_cnt = 0
    for y in ys:
        try:
            ny = one_hot(y, target_classes)
        except ValueError:
            error_cnt += 1
            if log_limit > 0:
                log_limit -= 1
                print(
                    "cant find target cls for y, skipping after 10 samples:", y, target_classes)
            ny = 0
        yy.append(ny)

    if error_cnt == len(ys):
        return None

    return yy


def split(xs, ys, train_rate):
    xs, ys = shuffle2(xs, ys)

    pos1 = int(len(xs) * train_rate)

    train_x, train_y = xs[0:pos1], ys[0:pos1]
    val_x, val_y = xs[pos1:], ys[pos1:]

    return train_x, train_y, val_x, val_y


def split3(xs, ys, keys, train_rate):
    xs, ys, keys = shuffle3(xs, ys, keys)

    pos1 = int(len(xs) * train_rate)
    # pos2 = int(len(x) * (train_rate + test_rate))

    train_x, train_y, train_keys = xs[0:pos1], ys[0:pos1], keys[0:pos1]
    # test_x, test_y = x[pos1:pos2], y[pos1:pos2]
    val_x, val_y, val_keys = xs[pos1:], ys[pos1:], keys[pos1:]

    return train_x, train_y, train_keys, val_x, val_y, val_keys


def split3_by_key(xs, ys, keys, train_rate, split_key, keep_min=None):
    # use split_key pattern to find all keys, assemble into key_set
    key_set = {}
    for k in keys:
        m = re.search(split_key, k)
        rk = m.group(0)
        if rk not in key_set:
            key_set[rk] = 0
        key_set[rk] += 1

    if keep_min is not None:
        for k in list(key_set.keys()):
            if key_set[k] < keep_min:
                del key_set[k]

    base_keys = list(key_set.keys())
    print("*** Found key groups (to split complete individuals):", len(base_keys))

    # split key set
    pos = int(len(base_keys) * train_rate)

    np.random.shuffle(base_keys)
    train_keys = base_keys[0:pos]
    train_key_set = {}
    for k in train_keys:
        train_key_set[k] = True

    val_keys = base_keys[pos + 1:]
    val_key_set = {}
    for k in val_keys:
        val_key_set[k] = True

    # use split key groups to assemble train and val data
    train_x, train_y, train_keys = [], [], []
    val_x, val_y, val_keys = [], [], []

    for idx, k in enumerate(keys):
        m = re.search(split_key, k)
        rk = m.group(0)
        if rk in train_key_set:
            train_x.append(xs[idx])
            train_y.append(ys[idx])
            train_keys.append(keys[idx])
        elif rk in val_key_set:
            val_x.append(xs[idx])
            val_y.append(ys[idx])
            val_keys.append(keys[idx])

    train_x, train_y, train_keys = shuffle3(train_x, train_y, train_keys)
    val_x, val_y, val_keys = shuffle3(val_x, val_y, val_keys)

    return train_x, train_y, train_keys, val_x, val_y, val_keys


def shuffle2(x, y):
    xy = list(zip(x, y))
    np.random.shuffle(xy)
    x, y = zip(*xy)
    return list(x), list(y)


def shuffle3(x, y, z):
    xyz = list(zip(x, y, z))
    np.random.shuffle(xyz)
    x, y, z = zip(*xyz)
    return list(x), list(y), list(z)


def count_matches(ys, pys, true_val):
    pp = []

    # for each sample
    for idx, y in enumerate(ys):
        # if it belongs to true_val (true label)
        if np.array_equal(y, true_val):
            # add it's prediction
            pp.append(pys[idx])

    if len(pp) == 0:
        return None

    return np.sum(pp, axis=0)


def acc(ys, pys):
    right = 0
    for idx, y in enumerate(ys):
        if np.array_equal(y, pys[idx]):
            right += 1

    # overall accuracy
    print("Accuracy: {:0.2f}%".format(100 * right / len(ys)))

    return right


def summarize(ys, pys, sample_target_classes):
    if len(sample_target_classes) > 20:
        print("Too many target classes")
        return

    # for each of the samples' category/class
    for tc in sample_target_classes:
        true_onehot = one_hot(tc, sample_target_classes)
        ac = count_matches(ys, pys, true_onehot)
        str = ""
        if ac is not None:
            percs = 100 * ac / ac.sum()
            for idx in range(len(ac)):
                str += "{:5.2f}% "
            str = str.format(*percs)
        print(str, '\t', ac, '\t', tc)


def summarize_compare(xs, ys, probys, pys, target_classes):
    accs = []
    for tc in target_classes:
        true_onehot = one_hot(tc, target_classes)
        ac = count_matches(ys, pys, true_onehot)
        accs.append(ac)
        percs = 100 * ac / ac.sum()
        str = ""
        for idx in range(len(ac)):
            str += "{:5.2f}% "
        print(tc, '\t', ac, '\t', str.format(*percs))


def dump_prediction(path, keys, probys, ys=None, append=True):
    if append:
        flags = "a"
    else:
        flags = "w"
    csvfile = open(path, flags)  # a to append
    csv_writer = csv.writer(csvfile, lineterminator='\n')

    # if shape len is 3, it's a stack of repeated prediction (bayes model)
    all_probys = None
    if len(probys.shape) == 3:
        all_probys = probys
        probys = probys[0]

    for idx, prob in enumerate(probys):
        key = keys[idx]

        result = [key]
        if ys:
            result += [ys[idx]]
        else:
            result += [""]
        result += prob.tolist()
        if all_probys is not None:
            for p_idx in range(1, len(all_probys)):
                result += all_probys[p_idx][idx].tolist()

        csv_writer.writerow(result)

        # p1 = prob[0]
        # p2 = prob[1]
        # csv_writer.writerow([p1, p2, key])
    csvfile.close()

    print("dumped predictions to:", path)


def convert_gray(xs):
    xs = list(xs)

    for idx, x in enumerate(xs):
        x = transform.rescale(x, 1.4, anti_aliasing=True)

        x = util.img_as_float(x)
        x = exposure.adjust_gamma(x, 0.5)

        x = color.rgb2gray(x)
        x = util.invert(x)
        x = util.img_as_ubyte(x)

        x = np.where(x[:, :] == 255, 0, x[:, :])

        m = round((x.shape[0] - 128) / 2)
        x = x[m:m + 128, m:m + 128]

        img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
        img[:, :, 0] = x
        img[:, :, 1] = x
        img[:, :, 2] = x

        xs[idx] = img

    return xs


def fix_channels(xs, shape):
    target_channels = shape[2]

    if len(xs[0].shape) == 3 and xs[0].shape[2] == target_channels:
        return xs

    if target_channels > 1:

        xx = []
        for x in xs:
            if len(x.shape) == 2:
                x = np.stack((x,) * target_channels, axis=-1)
            elif x.shape[2] == 1:
                x = np.stack((x[:, :, 0],) * target_channels, axis=-1)
            else:
                raise 'dont know how to handle'

            xx.append(x)
        xs = xx

    elif target_channels == 1:
        xx = []
        for x in xs:
            xx.append(x[:, :, 0])
        xs = xx

    return xs


def standardize_colors(xs):
    xs = xs.astype(np.float64)

    for idx, x in enumerate(xs):
        for ch in range(x.shape[2]):
            img = xs[idx, :, :, ch]

            xs[idx, :, :, ch] = (
                img - np.mean(img)) / (np.std(img) + 1e-6)

    return xs


def onehot(y_pred):
    return [get_pred(p) for p in y_pred]


def get_pred(probs):
    y = np.zeros((len(probs)))
    y[np.argmax(probs)] = 1
    return y
