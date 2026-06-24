python train.py \
  --data data/moana_xband.yaml \
  --weights yolov5s.pt \
  --imgsz 640 \
  --batch-size 16 \
  --single-cls \
  --evolve 30 \
  --project runs/evolve \
  --name evolve_xband
