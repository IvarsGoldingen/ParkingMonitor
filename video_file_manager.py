import os
import logging
import shutil

# Setup logging
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
# Console debug
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# File logger
file_handler = logging.FileHandler(os.path.join("logs", "video_file_mngr.log"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)


class FileManager():
    """
    FileManager
    Deletes oldest files from folder if it gets too large.
    """

    def get_folder_size(self, path, delete_files_smaller_than_gb):
        """
        Calculates folder size and deletes files that are too small. Small files are expected to be created when
        video recording stopped by error or Pycharm stop btn.
        :param path: Path to analyse
        :param delete_files_smaller_than_gb: Files msaller than this are deleted
        :return: size_gb, list_of_files
        """
        logger.debug(f'Getting size of and deleting filse smaller than {delete_files_smaller_than_gb}: from {path}')
        size_gb = 0.0
        list_of_files = []
        for dirpath, dirnames, filenames in os.walk(path, topdown=True):
            # Go through all directories
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                if not os.path.islink(file_path):
                    file_size_GB = float(os.path.getsize(file_path) / 1000000000)
                    if file_size_GB < delete_files_smaller_than_gb:
                        # Delete file too small
                        file_size_mb = file_size_GB * 1000
                        logger.debug(f"Deleting small file: {file_path} size: {file_size_mb} MB")
                        os.remove(file_path)
                    else:
                        # File size acceptable, add file to list of files and calculate total size
                        list_of_files.append(file_path)
                        size_gb += file_size_GB
        logger.debug(f'Number of files: {len(list_of_files)}')
        logger.debug(f"Size in GB: {size_gb}")
        return size_gb, list_of_files

    def limit_folder_size(self, path, delete_when_size_larger_than_gb, delete_size_gb, delete_files_smaller_than_gb):
        """
        Limits folder size according to parameters
        :param path: path to analyse
        :param delete_when_size_larger_than_gb:
        :param delete_size_gb: How much to delete if delete_when_size_larger_than_gb exceeded
        :param delete_files_smaller_than_gb:
        """
        logger.info(f"Checking folder for size: {path}")
        self.delete_empty_subfolders(path)
        folder_size, file_list = self.get_folder_size(path, delete_files_smaller_than_gb)
        if folder_size >= delete_when_size_larger_than_gb:
            # Folder exceeds size limit
            logger.info(
                f"Folder {folder_size} GB larger than size limit {delete_when_size_larger_than_gb} GB, deleting files.")
            deleted_gb = 0.0
            while deleted_gb < delete_size_gb and folder_size > 0.0:
                # Delete while folder has files in it and enough files have been deleted
                oldest_file = min(file_list, key=os.path.getctime)
                oldest_file_size_gb = float(os.path.getsize(oldest_file) / 1000000000)
                oldest_file_size_mb = oldest_file_size_gb * 1000
                logger.debug(f"Deleting oldest file which is {oldest_file} size: {oldest_file_size_mb} GB")
                os.remove(oldest_file)
                file_list.remove(oldest_file)
                folder_size -= oldest_file_size_gb
                deleted_gb += oldest_file_size_gb
                logger.debug(f"New folder size is {folder_size}")
            logger.info(f"Deleted {deleted_gb} GB of files. New folder size is {folder_size}")
        else:
            logger.info(f"Folder smaller than size limit, no action. {folder_size} GB")

    def delete_empty_subfolders(self, path):
        """
        Delete emty folders from path
        """
        for dirpath, dirnames, filenames in os.walk(path, topdown=True):
            for d in dirnames:
                file_path = os.path.join(dirpath, d)
                if len(os.listdir(file_path)) == 0:
                    # os.remove(file_path)
                    shutil.rmtree(file_path)
                    logger.debug(f"Deleted empty folder {file_path}")


if __name__ == '__main__':
    mngr = FileManager()
    mngr.get_folder_size("/home/pi/Documents/cam_recordings/2022_03_13")
