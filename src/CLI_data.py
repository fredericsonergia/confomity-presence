from detector_utils.VOC_form import rename_img
import fire


class Data_processor(object):
    def rename(self, path="../Data/Images/", start=1, is_ok=True):
        """
    renames images from a folder 
    Args:
    -path (str): the path of Images
    -start (int): the number from which the annotation begin
    -is_ok (bool): flag to specify the ok or ko images
    """
        rename_img(path, start, is_ok)


if __name__ == "__main__":
    fire.Fire(Data_processor)
