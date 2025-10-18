'use client';

import { useState, useEffect } from 'react';
import { 
  Calendar, 
  Download, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock,
  BarChart3,
  FileText,
  RefreshCw
} from 'lucide-react';
import { format, subDays } from 'date-fns';

// Types based on the API documentation
interface DailyStats {
  date: string;
  total_predictions: number;
  matches: number;
  no_matches: number;
  match_percentage: number;
  no_match_percentage: number;
  avg_processing_time_ms: number;
  avg_confidence: number;
}

interface Prediction {
  id: number;
  predicted_lot: string;
  predicted_expiry: string;
  is_match: boolean;
  match_status: string;
  detection_confidence: number;
  processing_time_ms: number;
  created_at: string;
  full_frame_path: string;
  crop_path: string;
  final_path: string;
}

interface PredictionsResponse {
  date: string;
  predictions: Prediction[];
  limit: number;
  offset: number;
  count: number;
}

interface SummaryData {
  today: DailyStats;
  yesterday: DailyStats;
  last_7_days: DailyStats[];
}

export default function ReportsPage() {
  const [selectedDate, setSelectedDate] = useState(format(new Date(), 'yyyy-MM-dd'));
  const [dailyStats, setDailyStats] = useState<DailyStats | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'daily' | 'predictions'>('dashboard');

  const API_BASE = 'http://localhost:8000';

  // Fetch summary data for dashboard
  const fetchSummary = async () => {
    try {
      const response = await fetch(`${API_BASE}/reports/summary`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setSummary(data);
    } catch (err: any) {
      setError(`Failed to fetch summary: ${err.message}`);
    }
  };

  // Fetch daily stats
  const fetchDailyStats = async (date: string) => {
    try {
      const response = await fetch(`${API_BASE}/reports/daily-stats?date=${date}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setDailyStats(data);
    } catch (err: any) {
      setError(`Failed to fetch daily stats: ${err.message}`);
    }
  };

  // Fetch predictions list
  const fetchPredictions = async (date: string, limit = 50) => {
    try {
      const response = await fetch(`${API_BASE}/reports/predictions?date=${date}&limit=${limit}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: PredictionsResponse = await response.json();
      setPredictions(data.predictions);
    } catch (err: any) {
      setError(`Failed to fetch predictions: ${err.message}`);
    }
  };

  // Download CSV export
  const downloadCSV = async (date: string) => {
    try {
      const response = await fetch(`${API_BASE}/reports/export-csv?date=${date}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lote_predictions_${date}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setError(`Failed to download CSV: ${err.message}`);
    }
  };

  // Load data based on active tab
  const loadData = async () => {
    setLoading(true);
    setError('');
    
    try {
      if (activeTab === 'dashboard') {
        await fetchSummary();
      } else if (activeTab === 'daily') {
        await fetchDailyStats(selectedDate);
      } else if (activeTab === 'predictions') {
        await fetchPredictions(selectedDate);
      }
    } finally {
      setLoading(false);
    }
  };

  // Load data when tab or date changes
  useEffect(() => {
    loadData();
  }, [activeTab, selectedDate]);

  // Stat Card Component
  const StatCard = ({ 
    title, 
    value, 
    subtitle, 
    icon: Icon, 
    trend, 
    color = 'blue' 
  }: {
    title: string;
    value: string | number;
    subtitle?: string;
    icon: any;
    trend?: 'up' | 'down' | 'neutral';
    color?: 'blue' | 'green' | 'red' | 'yellow';
  }) => {
    const colorClasses = {
      blue: 'bg-blue-50 border-blue-200 text-blue-900',
      green: 'bg-green-50 border-green-200 text-green-900',
      red: 'bg-red-50 border-red-200 text-red-900',
      yellow: 'bg-yellow-50 border-yellow-200 text-yellow-900',
    };

    const iconColors = {
      blue: 'text-blue-600',
      green: 'text-green-600',
      red: 'text-red-600',
      yellow: 'text-yellow-600',
    };

    return (
      <div className={`rounded-lg border p-4 ${colorClasses[color]}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium opacity-70">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
            {subtitle && <p className="text-xs opacity-60 mt-1">{subtitle}</p>}
          </div>
          <div className={`p-2 rounded-lg bg-white/50 ${iconColors[color]}`}>
            <Icon size={24} />
          </div>
        </div>
        {trend && (
          <div className="mt-2 flex items-center text-xs">
            {trend === 'up' && <TrendingUp size={12} className="mr-1 text-green-600" />}
            {trend === 'down' && <TrendingDown size={12} className="mr-1 text-red-600" />}
            {trend === 'neutral' && <Activity size={12} className="mr-1 text-gray-600" />}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <BarChart3 className="text-blue-600" size={28} />
              <h1 className="text-xl font-semibold text-gray-900">Lote Predictions Reports</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={loadData}
                disabled={loading}
                className="flex items-center space-x-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
                <span>Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
              { id: 'daily', label: 'Daily Report', icon: Calendar },
              { id: 'predictions', label: 'Predictions List', icon: FileText },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon size={16} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4">
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <XCircle className="text-red-400" size={20} />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && summary && (
          <div className="space-y-6">
            {/* Today's Overview */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Today's Overview</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                  title="Total Predictions"
                  value={summary.today.total_predictions}
                  icon={Activity}
                  color="blue"
                />
                <StatCard
                  title="Successful Matches"
                  value={summary.today.matches}
                  subtitle={`${summary.today.match_percentage}% success rate`}
                  icon={CheckCircle}
                  color="green"
                />
                <StatCard
                  title="Failed Matches"
                  value={summary.today.no_matches}
                  subtitle={`${summary.today.no_match_percentage}% failure rate`}
                  icon={XCircle}
                  color="red"
                />
                <StatCard
                  title="Avg Processing Time"
                  value={`${Math.round(summary.today.avg_processing_time_ms)}ms`}
                  subtitle={`Confidence: ${Math.round(summary.today.avg_confidence * 100)}%`}
                  icon={Clock}
                  color="yellow"
                />
              </div>
            </div>

            {/* Comparison with Yesterday */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Comparison with Yesterday</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Predictions</h3>
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold text-gray-900">
                      {summary.today.total_predictions}
                    </span>
                    <span className="text-sm text-gray-500">
                      vs {summary.yesterday.total_predictions} yesterday
                    </span>
                  </div>
                </div>
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Success Rate</h3>
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold text-green-600">
                      {summary.today.match_percentage}%
                    </span>
                    <span className="text-sm text-gray-500">
                      vs {summary.yesterday.match_percentage}% yesterday
                    </span>
                  </div>
                </div>
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Processing Time</h3>
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold text-blue-600">
                      {Math.round(summary.today.avg_processing_time_ms)}ms
                    </span>
                    <span className="text-sm text-gray-500">
                      vs {Math.round(summary.yesterday.avg_processing_time_ms)}ms yesterday
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Last 7 Days Trend */}
            {summary.last_7_days.length > 0 && (
              <div>
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Last 7 Days Trend</h2>
                <div className="bg-white rounded-lg border p-4">
                  <div className="space-y-3">
                    {summary.last_7_days.map((day) => (
                      <div key={day.date} className="flex items-center justify-between py-2 border-b last:border-b-0">
                        <div className="flex items-center space-x-3">
                          <span className="text-sm font-medium text-gray-900">{day.date}</span>
                          <span className="text-xs text-gray-500">{day.total_predictions} predictions</span>
                        </div>
                        <div className="flex items-center space-x-4">
                          <span className="text-sm font-medium text-green-600">
                            {day.match_percentage}% success
                          </span>
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-green-600 h-2 rounded-full"
                              style={{ width: `${day.match_percentage}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Daily Report Tab */}
        {activeTab === 'daily' && (
          <div className="space-y-6">
            {/* Date Picker */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Daily Report</h2>
              <div className="flex items-center space-x-4">
                <input
                  type="date"
                  value={selectedDate}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={() => downloadCSV(selectedDate)}
                  className="flex items-center space-x-2 px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 text-sm"
                >
                  <Download size={16} />
                  <span>Export CSV</span>
                </button>
              </div>
            </div>

            {/* Daily Stats */}
            {dailyStats && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                  title="Total Predictions"
                  value={dailyStats.total_predictions}
                  subtitle={`for ${dailyStats.date}`}
                  icon={Activity}
                  color="blue"
                />
                <StatCard
                  title="Successful Matches"
                  value={dailyStats.matches}
                  subtitle={`${dailyStats.match_percentage}% success rate`}
                  icon={CheckCircle}
                  color="green"
                />
                <StatCard
                  title="Failed Matches"
                  value={dailyStats.no_matches}
                  subtitle={`${dailyStats.no_match_percentage}% failure rate`}
                  icon={XCircle}
                  color="red"
                />
                <StatCard
                  title="Avg Processing Time"
                  value={`${Math.round(dailyStats.avg_processing_time_ms)}ms`}
                  subtitle={`Confidence: ${Math.round(dailyStats.avg_confidence * 100)}%`}
                  icon={Clock}
                  color="yellow"
                />
              </div>
            )}

            {dailyStats && dailyStats.total_predictions === 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <div className="flex">
                  <Activity className="text-yellow-400" size={20} />
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-yellow-800">No Data</h3>
                    <p className="text-sm text-yellow-700 mt-1">
                      No predictions found for {selectedDate}. Try selecting a different date.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Predictions List Tab */}
        {activeTab === 'predictions' && (
          <div className="space-y-6">
            {/* Header with Date Picker */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Predictions List</h2>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Predictions Table */}
            {predictions.length > 0 ? (
              <div className="bg-white rounded-lg border overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Time
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Predicted Lot
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Expiry Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Match Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Confidence
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Processing Time
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {predictions.map((prediction) => (
                        <tr key={prediction.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {format(new Date(prediction.created_at), 'HH:mm:ss')}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {prediction.predicted_lot || 'N/A'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {prediction.predicted_expiry || 'N/A'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                prediction.is_match
                                  ? 'bg-green-100 text-green-800'
                                  : 'bg-red-100 text-red-800'
                              }`}
                            >
                              {prediction.match_status}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {Math.round(prediction.detection_confidence * 100)}%
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {Math.round(prediction.processing_time_ms)}ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <div className="flex">
                  <FileText className="text-yellow-400" size={20} />
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-yellow-800">No Predictions</h3>
                    <p className="text-sm text-yellow-700 mt-1">
                      No predictions found for {selectedDate}. Try selecting a different date.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="animate-spin text-blue-600" size={32} />
            <span className="ml-3 text-lg text-gray-600">Loading...</span>
          </div>
        )}
      </div>
    </div>
  );
}
