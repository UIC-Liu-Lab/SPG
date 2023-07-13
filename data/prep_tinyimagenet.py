import glob
import os
import shutil

import pandas as pd


def prep():
    # check cwd
    cwd = os.getcwd().split('/')[-1]
    assert cwd == 'data', f'Re-run this program after you `cd` to `dat` directory.'

    # check tiny-imagenet-200
    dir_ti200 = './tiny-imagenet-200'
    assert os.path.exists(dir_ti200), f'Make sure you have placed it in {dir_ti200}'

    # read 200 classes
    list__cls = [file.split('/')[-1] for file in glob.glob(os.path.join(dir_ti200, 'train', 'n*'))]
    assert len(list__cls) == 200, f'Make sure it contains 200 classes.'

    # read val_annotation
    df = pd.read_table(os.path.join(dir_ti200, 'val', 'val_annotations.txt'),
                       names=['img', 'cls', '_1', '_2', '_3', '_4'])

    # re-organize dirs so that `torchvision.datasets.ImageFolder` can read them.
    for idx_cls, cls in enumerate(list__cls):
        print(f'Processsing {cls} [{idx_cls + 1}/{len(list__cls)}]')

        # [train]
        # remove `tiny-imagenet-200/train/<cls>/<cls>_boxes.txt`
        path_boxestxt = os.path.join(dir_ti200, 'train', cls, f'{cls}_boxes.txt')
        if os.path.exists(path_boxestxt):
            os.remove(path_boxestxt)
        # endif

        # move `tiny-imagenet-200/train/<cls>/images/*.JPEG` to `tiny-imagenet-200/train/<cls>`
        dir_train_img = os.path.join(dir_ti200, 'train', cls, 'images')
        if os.path.exists(dir_train_img):
            list__train_img = glob.glob(os.path.join(dir_train_img, f'{cls}*.JPEG'))
            for train_img in list__train_img:
                shutil.move(train_img, os.path.join(dir_ti200, 'train', cls))
            # endfor
            shutil.rmtree(dir_train_img)
        # endif

        # [test]
        # remove dir `tiny-imagenet-200/test/images/`
        dir_test_images = os.path.join(dir_ti200, 'test', 'images')
        if os.path.exists(dir_test_images):
            shutil.rmtree(dir_test_images)
        # endif

        # re-create dir for cls `tiny-imagenet-200/test/<cls>/`
        dir_test_cls = os.path.join(dir_ti200, 'test', cls)
        if os.path.exists(dir_test_cls):
            shutil.rmtree(dir_test_cls)
        # endif
        os.makedirs(dir_test_cls)

        # copy images from `tiny-imagenet-200/val/images/*.JPEG` to `tiny-imagenet-200/test/<cls>/`
        list__test_img = df[df['cls'] == cls]['img'].values.tolist()
        for test_img in list__test_img:
            shutil.copy(os.path.join(dir_ti200, 'val', 'images', test_img), dir_test_cls)
        # endfor
    # endfor


if __name__ == '__main__':
    prep()
