import os
import shutil


def delete_test_folders():
    current_directory = os.getcwd()

    for item in os.listdir(current_directory):
        item_path = os.path.join(current_directory, item)

        if os.path.isdir(item_path) and item.startswith("test_"):
            try:
                shutil.rmtree(item_path)
                print(f"Deleted: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")


if __name__ == "__main__":
    delete_test_folders()
