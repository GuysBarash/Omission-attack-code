import os
import shutil
import tqdm


def clear_folder(path, delete_if_exist=True):
    if os.path.exists(path) and delete_if_exist:
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    src_dir = r'C:\school\thesis\omission\CIFAR-10-images-master\test'
    trgt_dir = r'C:\school\thesis\omission\CIFAR_reduced\test'
    img_to_move = 100

    clear_folder(trgt_dir)
    categories = os.listdir(src_dir)
    for t_category in categories:
        src_t_cat = os.path.join(src_dir, t_category)
        trgt_t_cat = os.path.join(trgt_dir, t_category)
        clear_folder(trgt_t_cat)

        src_imgs = os.listdir(src_t_cat)[:img_to_move]
        for src_img in tqdm.tqdm(src_imgs, desc=f'{t_category:<15}', total=len(src_imgs)):
            src_img_path = os.path.join(src_t_cat, src_img)
            trgt_img_path = os.path.join(trgt_t_cat, src_img)

            shutil.copy(src_img_path, trgt_img_path)
