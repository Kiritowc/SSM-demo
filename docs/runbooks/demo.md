## 1 VLM 起

cd "/home/sunshink/ssdet VLM" && chmod +x vlm/scripts/start_vlm.sh vlm/scripts/stop_vlm.sh 2>/dev/null && ./vlm/scripts/start_vlm.sh

## 2 VLM 停

cd "/home/sunshink/ssdet VLM" && ./vlm/scripts/stop_vlm.sh

## 3 摄像头起

cd "/home/sunshink/ssdet VLM" && chmod +x camera/scripts/start_camera_with_status.sh camera/scripts/start_camera.sh camera/scripts/stop_camera.sh 2>/dev/null && ./camera/scripts/start_camera_with_status.sh

## 4 摄像头停

cd "/home/sunshink/ssdet VLM" && ./camera/scripts/stop_camera.sh

## 5 网页

http://127.0.0.1:9080/

http://172.16.16.165:9080/

## 6 训练

cd "/home/sunshink/ssdet VLM" && conda activate ssdet && python -m cv.train

## 7 SSH

ssh -N -L 18080:127.0.0.1:9080 sunshink@172.16.16.165
