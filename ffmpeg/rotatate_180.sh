echo "Rotating video"
video_in=Medicinas_2025_10_08_11_39_11.mp4
video_out=Medicinas_rotated_180.mp4

ffmpeg -i $video_in -vf "transpose=2,transpose=2" -c:a copy $video_out

echo "Video rotated"