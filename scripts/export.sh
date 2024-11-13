python src/onnx_export.py \
    --checkpoint_path 'weights/best.ckpt' \
    --batch_size 1 \
    --simplify \
    --cleanup \
    --dynamic-batch
