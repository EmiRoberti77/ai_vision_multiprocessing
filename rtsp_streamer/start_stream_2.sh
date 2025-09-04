echo "Running start_stream_1.sh"

IP=$(ip route get 8.8.8.8 | awk '{print $7}')
echo "IP: $IP"

ffmpeg -re -stream_loop -1 -i videos/crowd.mp4 \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -c:a aac -ar 44100 -b:a 128k -ac 2 \
  -f rtsp -rtsp_transport tcp rtsp://$IP:8554/mystream_2

echo "Stream 1 started"