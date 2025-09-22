#!/usr/bin/env python3
"""
Streamlit Detection Viewer Application

This application reads the OAIX database and displays detection events
with their corresponding images for manual verification and testing.
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
from PIL import Image
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="OAIX Detection Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path
DB_PATH = "/mnt/c/code/AI_Vision/multiprocessing/oaix.db"
OCR_BASE_PATH = "/mnt/c/code/AI_Vision/multiprocessing"

@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_detection_data():
    """Load detection data from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT 
            id,
            camera_name,
            lot,
            expiry,
            all_text,
            created_at,
            mime,
            image_path
        FROM webhook_events 
        ORDER BY created_at DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert created_at to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_app_logs():
    """Load application logs from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT 
            id,
            log_code,
            level,
            message,
            created_at
        FROM app_logs 
        ORDER BY created_at DESC
        LIMIT 100
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert created_at to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
    except Exception as e:
        st.error(f"Error loading logs from database: {e}")
        return pd.DataFrame()

def display_image(image_path):
    """Display an image from the given path."""
    if not image_path:
        st.warning("No image path provided")
        return
    if image_path == "_EMPTY":
        st.info("No saved OCR image for this event")
        return
    
    try:
        # Handle both absolute and relative paths
        if not os.path.isabs(image_path):
            full_path = os.path.join(OCR_BASE_PATH, image_path)
        else:
            full_path = image_path
            
        if os.path.exists(full_path):
            image = Image.open(full_path)
            st.image(image, caption=f"Detection Image: {os.path.basename(full_path)}", width='stretch')
        else:
            st.error(f"Image not found: {full_path}")
            st.info(f"Looking for: {full_path}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

def main():
    """Main Streamlit application."""
    
    st.title("🔍 OAIX Detection Viewer")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Filters & Options")
    
    # Load data
    df = load_detection_data()
    
    if df.empty:
        st.warning("No detection data found in the database.")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        return
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Camera filter
    cameras = ["All"] + list(df['camera_name'].unique())
    selected_camera = st.sidebar.selectbox("Camera:", cameras)
    
    # Date range filter
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    
    # Calculate default range - use actual date range or 7 days, whichever is smaller
    default_start = max(min_date, max_date - timedelta(days=7))
    
    date_range = st.sidebar.date_input(
        "Date Range:",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Text content filter
    text_filter = st.sidebar.text_input("Search in detected text:", "")
    
    # Lot/Expiry filter
    lot_filter = st.sidebar.text_input("Lot number contains:", "")
    expiry_filter = st.sidebar.text_input("Expiry contains:", "")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_camera != "All":
        filtered_df = filtered_df[filtered_df['camera_name'] == selected_camera]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= start_date) & 
            (filtered_df['created_at'].dt.date <= end_date)
        ]
    
    if text_filter:
        filtered_df = filtered_df[
            filtered_df['all_text'].str.contains(text_filter, case=False, na=False)
        ]
    
    if lot_filter:
        filtered_df = filtered_df[
            filtered_df['lot'].str.contains(lot_filter, case=False, na=False)
        ]
    
    if expiry_filter:
        filtered_df = filtered_df[
            filtered_df['expiry'].str.contains(expiry_filter, case=False, na=False)
        ]
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", len(filtered_df))
    
    with col2:
        unique_cameras = filtered_df['camera_name'].nunique()
        st.metric("Cameras", unique_cameras)
    
    with col3:
        detections_with_lot = len(filtered_df[filtered_df['lot'].notna() & (filtered_df['lot'] != "")])
        st.metric("With Lot Info", detections_with_lot)
    
    with col4:
        detections_with_expiry = len(filtered_df[filtered_df['expiry'].notna() & (filtered_df['expiry'] != "")])
        st.metric("With Expiry Info", detections_with_expiry)
    
    # Charts
    if len(filtered_df) > 0:
        st.subheader("📊 Detection Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Detections over time
            daily_counts = filtered_df.groupby(filtered_df['created_at'].dt.date).size().reset_index()
            daily_counts.columns = ['Date', 'Count']
            
            fig_timeline = px.line(
                daily_counts, 
                x='Date', 
                y='Count',
                title='Detections Over Time',
                markers=True
            )
            st.plotly_chart(fig_timeline, width='stretch')
        
        with col2:
            # Detections by camera
            camera_counts = filtered_df['camera_name'].value_counts().reset_index()
            camera_counts.columns = ['Camera', 'Count']
            
            fig_cameras = px.pie(
                camera_counts,
                values='Count',
                names='Camera',
                title='Detections by Camera'
            )
            st.plotly_chart(fig_cameras, width='stretch')
    
    # Display mode selection
    st.subheader("📋 Detection Data")
    
    display_mode = st.radio(
        "Display Mode:",
        ["Table View", "Card View", "Image Gallery"],
        horizontal=True
    )
    
    if display_mode == "Table View":
        # Table view with selectable rows
        selected_indices = st.dataframe(
            filtered_df[['id', 'camera_name', 'lot', 'expiry', 'all_text', 'created_at']],
            width='stretch',
            selection_mode="single-row",
            on_select="rerun"
        )
        
        # Show image for selected row
        if selected_indices['selection']['rows']:
            selected_idx = selected_indices['selection']['rows'][0]
            selected_row = filtered_df.iloc[selected_idx]
            
            st.subheader(f"🖼️ Detection Details - ID: {selected_row['id']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_image(selected_row['image_path'])
            
            with col2:
                st.write("**Detection Information:**")
                st.write(f"**Camera:** {selected_row['camera_name']}")
                st.write(f"**Date:** {selected_row['created_at']}")
                st.write(f"**Lot:** {selected_row['lot'] or 'N/A'}")
                st.write(f"**Expiry:** {selected_row['expiry'] or 'N/A'}")
                st.write(f"**MIME:** {selected_row['mime']}")
                st.write("**Detected Text:**")
                st.text_area("", selected_row['all_text'] or "No text detected", height=150, disabled=True)
    
    elif display_mode == "Card View":
        # Card view - show each detection as a card
        for idx, row in filtered_df.iterrows():
            with st.expander(f"🔍 Detection {row['id']} - {row['camera_name']} - {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    display_image(row['image_path'])
                
                with col2:
                    st.write("**Detection Information:**")
                    st.write(f"**ID:** {row['id']}")
                    st.write(f"**Camera:** {row['camera_name']}")
                    st.write(f"**Date:** {row['created_at']}")
                    st.write(f"**Lot:** {row['lot'] or 'N/A'}")
                    st.write(f"**Expiry:** {row['expiry'] or 'N/A'}")
                    st.write(f"**MIME:** {row['mime']}")
                    st.write("**Detected Text:**")
                    st.text(row['all_text'] or "No text detected")
    
    elif display_mode == "Image Gallery":
        # Image gallery view
        cols_per_row = 3
        for i in range(0, len(filtered_df), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(filtered_df):
                    row = filtered_df.iloc[i + j]
                    
                    with col:
                        st.write(f"**ID {row['id']} - {row['camera_name']}**")
                        display_image(row['image_path'])
                        st.caption(f"Date: {row['created_at'].strftime('%Y-%m-%d %H:%M')}")
                        st.caption(f"Lot: {row['lot'] or 'N/A'} | Expiry: {row['expiry'] or 'N/A'}")
                        if row['all_text']:
                            st.caption(f"Text: {row['all_text'][:50]}...")
    
    # Refresh button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 Refresh Data", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    # App logs section
    with st.expander("📋 Application Logs"):
        logs_df = load_app_logs()
        if not logs_df.empty:
            st.dataframe(logs_df, width='stretch')
        else:
            st.info("No logs available")

if __name__ == "__main__":
    main()
