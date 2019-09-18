from data_preprocessing import data_preprocessing

if __name__ == '__main__':
    step = 512
    patch_size = 512
    scale_range = 0.6
    stain_channels = ["h", "s", "v"]
    aug_num = 2

    p = data_preprocessing(
        step=step,
        patch_size=patch_size,
        scale_range=scale_range,
        stain_channels=stain_channels,
        aug_num=aug_num
    )
    img_path_list = ["E://biototem//iciar2018-challenge//pics//A08_test.png", ]
    p.cut_data(img_path_list, save_dir="../pics/test")
