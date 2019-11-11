import os


def walk_to_level(path, level=None):
    if level is None:
        yield from os.walk(path)
        return

    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            # When some directory on or below the desired level is found, all
            # of its subdirs are removed from the list of subdirs to search next.
            # So they won't be walked.
            del dirs[:]


def list_files(path, valid_exts=None, level=None, contains=None):
    # Loop over the input directory structure
    for (root_dir, dir_names, filenames) in walk_to_level(path, level):
        for filename in sorted(filenames):
            # ignore the file if not contains the string
            if contains is not None and contains not in filename:
                continue

            # Determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            if valid_exts and ext.endswith(valid_exts):
                # Construct the path to the file and yield it
                file = os.path.join(root_dir, filename)
                yield file
