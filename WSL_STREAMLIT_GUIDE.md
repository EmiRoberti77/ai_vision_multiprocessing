# 🔍 OAIX Detection Viewer - WSL Setup Guide

## Current Status ✅

The Streamlit application is successfully running! Here's what we verified:

- ✅ Database connection working
- ✅ 17 detection events found
- ✅ Image paths are accessible
- ✅ Streamlit server is running on port 8501
- ✅ Date range filter issue fixed (was causing API exception)

## How to Access the Application

### Option 1: Web Browser (Recommended)
1. Open your web browser (Chrome, Firefox, Edge)
2. Navigate to one of these URLs:
   - `http://localhost:8501`
   - `http://127.0.0.1:8501`
   - `http://0.0.0.0:8501`

### Option 2: Windows Host Access
If you're accessing from Windows (not WSL terminal):
1. Get your WSL IP address:
   ```bash
   hostname -I | awk '{print $1}'
   ```
2. Open browser and go to: `http://[WSL_IP]:8501`

## Starting the Application

### Method 1: Using the Script
```bash
cd /mnt/c/code/AI_Vision/multiprocessing
./run_streamlit.sh
```

### Method 2: Manual Start
```bash
source /home/emi/ai_env311/bin/activate
cd /mnt/c/code/AI_Vision/multiprocessing
streamlit run streamlit_detection_viewer.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

## Stopping the Application

### If running in background:
```bash
pkill -f streamlit
```

### If running in terminal:
Press `Ctrl+C`

## Testing the Database

Run the test script to verify everything is working:
```bash
source /home/emi/ai_env311/bin/activate
python test_db_connection.py
```

## Current Detection Data

Your database contains:
- **17 detection events** from camera `channel_1`
- **8 application logs**
- **Images stored in**: `/mnt/c/code/AI_Vision/multiprocessing/ocr/`

## Application Features

The web interface provides:

### 📊 Dashboard
- Total detections count: **17**
- Active cameras: **1** (channel_1)
- Detection timeline charts
- Camera distribution

### 🔍 Filtering Options
- Filter by camera name
- Date range selection
- Search in detected text
- Filter by lot/expiry information

### 📋 Display Modes
1. **Table View**: Interactive table with image preview
2. **Card View**: Expandable cards for each detection
3. **Image Gallery**: Grid view of all detection images

### Recent Detections Found
1. Text: "0 5 2026" (ID: 17)
2. Text: "699546"07087 Parti No:: Son Kull: Ta:" (ID: 16)
3. Text: "695540 0708" (ID: 15)
4. And more...

## Troubleshooting

### If the application doesn't load:
1. Check if Streamlit is running: `ps aux | grep streamlit`
2. Check the port: `netstat -tulpn | grep :8501`
3. Restart the application: `./run_streamlit.sh`

### If images don't display:
- All image paths have been verified as accessible
- Images are stored in: `/mnt/c/code/AI_Vision/multiprocessing/ocr/`

### WSL-Specific Issues:
- Use `0.0.0.0` as server address for better WSL compatibility
- Enable headless mode to prevent browser auto-launch issues
- Access via localhost or WSL IP address

## URLs to Try

1. `http://localhost:8501` ⭐ (Primary)
2. `http://127.0.0.1:8501`
3. `http://0.0.0.0:8501`

## Next Steps

Once you access the web interface, you can:
1. Review all 17 detection events
2. Verify OCR text extraction accuracy
3. Check lot and expiry detection
4. Export data for analysis
5. Monitor real-time detection performance

**The application is ready to use! 🚀**
