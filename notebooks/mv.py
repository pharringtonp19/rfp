import shutil

# source and destination directory paths
src_dir = '/Users/patrickpower/Documents/GitHub/evictions/styles'
dst_dir = '/Users/patrickpower/Documents/GitHub/rfp/styles'

# copy the directory and its contents recursively
shutil.copytree(src_dir, dst_dir)