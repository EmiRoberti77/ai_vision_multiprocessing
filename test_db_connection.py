#!/usr/bin/env python3
"""
Test script to verify database connection and check for data
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "/mnt/c/code/AI_Vision/multiprocessing/oaix.db"

def test_database():
    print("🔍 Testing OAIX Database Connection")
    print("=" * 50)
    
    # Check if database file exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file not found: {DB_PATH}")
        return False
    
    print(f"✅ Database file found: {DB_PATH}")
    print(f"📂 File size: {os.path.getsize(DB_PATH)} bytes")
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Tables found: {[table[0] for table in tables]}")
        
        # Check webhook_events table
        if ('webhook_events',) in tables:
            cursor.execute("SELECT COUNT(*) FROM webhook_events")
            webhook_count = cursor.fetchone()[0]
            print(f"📊 Webhook events count: {webhook_count}")
            
            if webhook_count > 0:
                cursor.execute("SELECT * FROM webhook_events ORDER BY created_at DESC LIMIT 5")
                recent_events = cursor.fetchall()
                print("\n🔍 Recent webhook events:")
                for i, event in enumerate(recent_events, 1):
                    print(f"  {i}. ID: {event[0]}, Camera: {event[1]}, Text: {event[4][:50] if event[4] else 'None'}...")
        
        # Check app_logs table
        if ('app_logs',) in tables:
            cursor.execute("SELECT COUNT(*) FROM app_logs")
            logs_count = cursor.fetchone()[0]
            print(f"📝 App logs count: {logs_count}")
            
            if logs_count > 0:
                cursor.execute("SELECT * FROM app_logs ORDER BY created_at DESC LIMIT 3")
                recent_logs = cursor.fetchall()
                print("\n📝 Recent app logs:")
                for i, log in enumerate(recent_logs, 1):
                    print(f"  {i}. Level: {log[2]}, Message: {log[3][:60] if log[3] else 'None'}...")
        
        conn.close()
        print("\n✅ Database connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_image_paths():
    print("\n🖼️  Testing Image Paths")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT image_path FROM webhook_events WHERE image_path IS NOT NULL LIMIT 5")
        image_paths = cursor.fetchall()
        
        if not image_paths:
            print("ℹ️  No image paths found in database")
            return
        
        print(f"🔍 Checking {len(image_paths)} image path(s):")
        
        for i, (image_path,) in enumerate(image_paths, 1):
            if image_path:
                # Check if absolute path exists
                if os.path.isabs(image_path):
                    full_path = image_path
                else:
                    # Try relative to project root
                    full_path = os.path.join("/mnt/c/code/AI_Vision/multiprocessing", image_path)
                
                exists = os.path.exists(full_path)
                status = "✅" if exists else "❌"
                print(f"  {i}. {status} {full_path}")
                
                if not exists and not os.path.isabs(image_path):
                    # Try the path as stored
                    alt_exists = os.path.exists(image_path)
                    if alt_exists:
                        print(f"     ✅ Found at original path: {image_path}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking image paths: {e}")

if __name__ == "__main__":
    test_database()
    check_image_paths()
    print(f"\n⏰ Test completed at: {datetime.now()}")
