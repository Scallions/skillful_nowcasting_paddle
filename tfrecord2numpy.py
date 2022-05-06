import tensorflow as tf
import numpy as np
import os

_FEATURES = {name: tf.io.FixedLenFeature([], dtype)
             for name, dtype in [
                 ("radar", tf.string), ("sample_prob", tf.float32),
                 ("osgb_extent_top", tf.int64), ("osgb_extent_left", tf.int64),
                 ("osgb_extent_bottom", tf.int64), ("osgb_extent_right", tf.int64),
                 ("end_time_timestamp", tf.int64),
             ]}

_SHAPE_BY_SPLIT_VARIANT = {
    ("train", "random_crops_256"): (24, 256, 256, 1),
    ("valid", "subsampled_tiles_256_20min_stride"): (24, 256, 256, 1),
    ("test", "full_frame_20min_stride"): (24, 1536, 1280, 1),
    ("test", "subsampled_overlapping_padded_tiles_512_20min_stride"): (24, 512, 512, 1),
}

_MM_PER_HOUR_INCREMENT = 1 / 32.
_MAX_MM_PER_HOUR = 128.
_INT16_MASK_VALUE = -1


def parse_and_preprocess_row(row, split="test", variant="full_frame_20min_stride"):
    result = tf.io.parse_example(row, _FEATURES)
    shape = _SHAPE_BY_SPLIT_VARIANT[(split, variant)]
    radar_bytes = result.pop("radar")
    radar_int16 = tf.reshape(tf.io.decode_raw(radar_bytes, tf.int16), shape)
    mask = tf.not_equal(radar_int16, _INT16_MASK_VALUE)
    radar = tf.cast(radar_int16, tf.float32) * _MM_PER_HOUR_INCREMENT
    radar = tf.clip_by_value(
        radar, _INT16_MASK_VALUE * _MM_PER_HOUR_INCREMENT, _MAX_MM_PER_HOUR)
    result["radar_frames"] = radar
    result["radar_mask"] = mask
    return result


def writeNpy(root, name, savePath):
    """
    :param root: save position file root
    :param name: file raw name
    :param savePath: save npy position
    """
    data = tf.data.TFRecordDataset(root, compression_type='GZIP')
    namePrefix = name.split('.')[0]
    parse_dataset = data.map(parse_and_preprocess_row)
    count = 0
    for i in parse_dataset:
        ett = i['end_time_timestamp'].numpy()
        oeb = i['osgb_extent_bottom'].numpy()
        oel = i['osgb_extent_left'].numpy()
        oet = i['osgb_extent_top'].numpy()
        sp = i['sample_prob'].numpy()
        rf = i['radar_frames'].numpy()
        rm = i['radar_mask'].numpy()
        dataDict = {'end_time_timestamp': ett, 'osgb_extent_bottom': oeb, 'osgb_extent_left': oel,
                    'osgb_extent_top': oet, 'sample_prob': sp, 'radar_frames': rf, 'radar_mask': rm}
        np.save(os.path.join(savePath, namePrefix + '_' + str(count) + '.npy'), dataDict)
        count += 1


def loadNpy(path):
    data = np.load(path)
    return data


if __name__ == '__main__':
    baseDir = r'E:\test111'
    savePath = r'E:\test222'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    fileName = os.listdir(baseDir)
    for name in fileName:
        writeNpy(os.path.join(baseDir, name), name, savePath)
    print('convert tfrecord to numpy done!')
