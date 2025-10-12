echo "Running start_stream_4.sh"

IP=$(ip route get 8.8.8.8 | awk '{print $7}')
echo "IP: $IP"

ffmpeg -re -stream_loop -1 -i videos/Medicinas_rotated_180_1.mp4 \
  -vf "scale=1920:1080" \
  -c:v libx264 -preset medium -profile:v main -level 4.0 \
  -pix_fmt yuv420p -r 30 -g 60 -keyint_min 60 -sc_threshold 0 \
  -b:v 2000k -maxrate 2500k -bufsize 4000k \
  -x264-params "nal-hrd=cbr:force-cfr=1" \
  -c:a aac -ar 44100 -b:a 128k -ac 2 \
  -f rtsp -rtsp_transport tcp \
  rtsp://$IP:8554/mystream_4

echo "Stream 4 started"