# Set GPU.
export CUDA_VISIBLE_DEVICES="0" 

export DATASET_PATH=PATH_TO_DATASET
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64 
export colmap=PATH_TO_COLMAP_INSTALL/bin/colmap
export mask=PATH_TO_GRABBER_MASK

for filename in $DATASET_PATH/*; do
	echo $filename
	$colmap database_creator --database_path $filename/db.db
	$colmap feature_extractor --database_path $filename/db.db --image_path $filename/images --ImageReader.camera_mask_path $mask --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1 
	colmap exhaustive_matcher --database_path $filename/db.db --SiftMatching.use_gpu 1
	mkdir -p $filename/sparse
	$colmap mapper --database_path $filename/db.db --image_path $filename/images --output_path $filename/sparse 
	for f in $filename/sparse/*; do
	    $colmap model_converter --input_path $f --output_path $f --output_type TXT
	    $colmap model_converter --input_path $f --output_path $f/export.ply --output_type PLY
	done
done

