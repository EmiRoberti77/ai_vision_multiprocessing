# Medicine OCR Frontend - Features & Usage

## Overview

This Next.js application provides a comprehensive interface for the Medicine OCR system, featuring real-time processing and detailed reporting capabilities.

## Features

### 🎥 Live Processing Page (`/`)
- **Real-time OCR Processing**: Submit frames for medicine label detection
- **Visual Results**: View full frame, cropped ROI, and final annotated images
- **Detection Metrics**: See confidence scores, bounding boxes, and class IDs
- **Lote Validation**: Real-time database matching with color-coded status
- **Performance Monitoring**: Processing time and detection statistics

### 📊 Reports Dashboard (`/reports`)

#### Dashboard Tab
- **Today's Overview**: Total predictions, success rate, processing time
- **Yesterday Comparison**: Side-by-side performance metrics
- **7-Day Trend**: Historical success rates with visual progress bars
- **Real-time Stats**: Auto-refreshing data with manual refresh option

#### Daily Report Tab
- **Date Selection**: Pick any date for detailed analysis
- **Comprehensive Stats**: 
  - Total predictions for the day
  - Success/failure breakdown with percentages
  - Average processing time and confidence scores
- **CSV Export**: Download detailed reports for external analysis
- **No Data Handling**: Clear messaging when no data exists

#### Predictions List Tab
- **Detailed Table View**: All predictions for a selected date
- **Sortable Columns**: Time, lot number, expiry, match status, confidence
- **Status Indicators**: Color-coded match/no-match badges
- **Pagination Ready**: Built for handling large datasets
- **Time Formatting**: Human-readable timestamps

## API Integration

The frontend consumes the following API endpoints:

### Dashboard Data
```javascript
GET /reports/summary
// Returns today, yesterday, and 7-day trend data
```

### Daily Statistics
```javascript
GET /reports/daily-stats?date=YYYY-MM-DD
// Returns comprehensive stats for a specific date
```

### Predictions List
```javascript
GET /reports/predictions?date=YYYY-MM-DD&limit=50&offset=0
// Returns paginated list of predictions
```

### CSV Export
```javascript
GET /reports/export-csv?date=YYYY-MM-DD
// Downloads CSV file with all predictions for the date
```

## UI Components

### StatCard Component
Reusable card component for displaying metrics with:
- **Color Themes**: Blue, green, red, yellow
- **Icons**: Lucide React icons for visual context
- **Trend Indicators**: Up/down/neutral trend arrows
- **Flexible Content**: Title, value, subtitle support

### Navigation Component
- **Responsive Design**: Desktop and mobile-friendly
- **Active State**: Highlights current page
- **Icon Integration**: Visual navigation cues
- **Accessibility**: Proper ARIA labels and keyboard navigation

## Styling & Design

### Tailwind CSS Classes
- **Color Palette**: 
  - Blue: Primary actions and info
  - Green: Success states and matches
  - Red: Errors and failed matches
  - Yellow: Warnings and processing metrics
- **Responsive Grid**: Mobile-first responsive design
- **Consistent Spacing**: Standardized padding and margins
- **Typography**: Clear hierarchy with proper font weights

### Visual Feedback
- **Loading States**: Spinning refresh icons
- **Error Messages**: Clear error boundaries with actionable messages
- **Empty States**: Helpful messaging when no data is available
- **Status Badges**: Color-coded success/failure indicators

## Data Flow

### Live Processing
1. User submits processing request
2. API returns detection results with lote match status
3. Images are displayed with overlay information
4. Results are automatically saved to database

### Reporting
1. User selects date/tab in reports interface
2. Frontend fetches data from appropriate API endpoint
3. Data is formatted and displayed in cards/tables
4. User can export data or navigate to different views

## Error Handling

### API Errors
- **Network Issues**: Clear error messages with retry options
- **HTTP Errors**: Status code and message display
- **Data Validation**: Graceful handling of malformed responses

### User Experience
- **Loading States**: Visual feedback during data fetching
- **Empty States**: Helpful guidance when no data exists
- **Fallback Values**: Default values for missing data fields

## Performance Optimizations

### React Optimizations
- **Client-side Rendering**: Fast page transitions
- **State Management**: Efficient useState hooks
- **Effect Dependencies**: Proper useEffect dependency arrays
- **Component Memoization**: Ready for React.memo if needed

### Data Loading
- **Conditional Fetching**: Only load data when needed
- **Error Boundaries**: Prevent crashes from API failures
- **Responsive Design**: Optimized for all screen sizes

## Development Setup

### Prerequisites
```bash
Node.js 18+
npm or yarn
```

### Installation
```bash
cd next_test_client/med_client
npm install
npm run dev
```

### Dependencies
- **Next.js 15**: React framework
- **Tailwind CSS 4**: Utility-first styling
- **Lucide React**: Icon library
- **date-fns**: Date formatting utilities
- **TypeScript**: Type safety

## Usage Examples

### Viewing Today's Performance
1. Navigate to `/reports`
2. Dashboard tab shows today's overview
3. Compare with yesterday's performance
4. View 7-day trend for context

### Analyzing Specific Date
1. Go to "Daily Report" tab
2. Select date using date picker
3. View comprehensive statistics
4. Export CSV for detailed analysis

### Monitoring Live Processing
1. Go to main page (`/`)
2. Submit processing requests
3. View real-time results
4. Check lote match status
5. Access saved artifacts

### Exporting Data
1. Select desired date in Daily Report
2. Click "Export CSV" button
3. File downloads automatically
4. Open in Excel/Google Sheets for analysis

## Future Enhancements

### Potential Features
- **Date Range Reports**: Multi-day analysis
- **Charts & Graphs**: Visual trend analysis
- **Real-time Updates**: WebSocket integration
- **User Management**: Authentication and roles
- **Advanced Filtering**: Search and filter predictions
- **Batch Operations**: Bulk data management

### Performance Improvements
- **Data Caching**: Client-side caching for faster loads
- **Virtual Scrolling**: Handle large datasets efficiently
- **Progressive Loading**: Load data as needed
- **Offline Support**: Service worker integration

## Troubleshooting

### Common Issues
1. **API Connection**: Ensure backend is running on port 8000
2. **CORS Errors**: Check CORS configuration in FastAPI
3. **Date Format**: Use YYYY-MM-DD format for date parameters
4. **Empty Data**: Check if predictions exist for selected date

### Debug Tips
- Check browser console for API errors
- Verify API endpoints are accessible
- Ensure date formats match API expectations
- Check network tab for failed requests

This frontend provides a complete solution for monitoring and analyzing your medicine OCR system's performance with a modern, responsive interface.
